import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, num_encoding_functions=6, include_input=True, log_sampling=True):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.log_sampling = log_sampling
        if self.num_encoding_functions <= 0:
            self.frequency_bands = None
            self.output_dim = input_dims if include_input else 0
            return

        if self.log_sampling:
            bands = 2.0 ** torch.linspace(0.0, self.num_encoding_functions - 1, self.num_encoding_functions)
        else:
            bands = torch.linspace(1.0, 2.0 ** (self.num_encoding_functions - 1), self.num_encoding_functions)
        self.register_buffer('frequency_bands', bands, persistent=False)

        # Calculate output dim here
        self.output_dim = 0
        if self.include_input:
            # Placeholder dims, will be updated in get_output_dim based on actual input
            self.output_dim += 2
        if self.frequency_bands is not None:
             # Placeholder dims, will be updated in get_output_dim based on actual input
            self.output_dim += self.num_encoding_functions * 2 * 2

    def forward(self, x):
        # x shape: (B, N, InputDims) e.g., (B, N, 2) for coordinates
        outputs = []
        if self.include_input:
            outputs.append(x)

        if self.frequency_bands is not None:
            bands = self.frequency_bands.to(x.device)
            # Apply encoding to each input dimension separately then concat? Or apply to vector?
            # Original implementation likely applies element-wise then concatenates features
            # Ensure shape compatibility: bands need broadcasting or explicit expansion
            # x * freq -> (B, N, InputDims) * (NumFunc) -> needs broadcasting or reshape
            # Let's match common NeRF implementations: Expand dims for broadcasting
            # x -> (B, N, InputDims, 1)
            # bands -> (1, 1, 1, NumFunc)
            # encoded = x.unsqueeze(-1) * bands.view(1, 1, 1, -1) * math.pi # (B, N, InputDims, NumFunc)
            # outputs.append(torch.sin(encoded))
            # outputs.append(torch.cos(encoded))
            # Or simpler if applied directly:
            for freq in bands:
                 outputs.extend([torch.sin(x * freq * math.pi), torch.cos(x * freq * math.pi)])

        if not outputs: return x # Return input if no encoding applied
        # Concatenate along the last dimension (features)
        # If using extend:
        result = torch.cat(outputs, dim=-1)
        # If using the broadcast method above:
        # result = torch.cat(outputs, dim=-1).view(x.shape[0], x.shape[1], -1) # Reshape (B, N, D*F*2 [+ D])

        return result

    def get_output_dim(self, input_dims=2):
        out_dim = 0
        if self.include_input:
            out_dim += input_dims
        if hasattr(self, 'frequency_bands') and self.frequency_bands is not None:
            out_dim += self.num_encoding_functions * 2 * input_dims
        # Ensure output dim is at least input dim if input is included
        return max(out_dim, input_dims if self.include_input else 0)


# --- CBAM Modules ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Use 1x1 Conv for FC layers adaptation
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_att = self.conv1(x_cat)
        return self.sigmoid(x_att)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x_out = x * self.ca(x) # Apply Channel Attention
        x_out = x_out * self.sa(x_out) # Apply Spatial Attention
        return x_out

# --- Depthwise Separable Convolution ---
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, norm_groups=4):
        super().__init__()
        use_norm = norm_groups > 0 and in_channels >= norm_groups and out_channels >= norm_groups

        # Depthwise Convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.norm_dw = nn.GroupNorm(num_groups=norm_groups, num_channels=in_channels) if use_norm else nn.Identity()
        # No ReLU after depthwise typically

        # Pointwise Convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.norm_pw = nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels) if use_norm else nn.Identity()
        self.relu_pw = nn.ReLU(inplace=True) # Activation after pointwise

    def forward(self, x):
        x = self.depthwise(x)
        x = self.norm_dw(x)
        # No intermediate ReLU here
        x = self.pointwise(x)
        x = self.norm_pw(x)
        x = self.relu_pw(x)
        return x

# --- Context CNN (Using Depthwise Separable) ---
class ContextCNN(nn.Module):
    def __init__(self, in_channels=3, out_features=32, norm_groups=4):
        super().__init__()
        print(f"Using Depthwise Separable CNN with GroupNorm (groups={norm_groups})")
        if norm_groups <= 0:
            print("Warning: norm_groups <= 0, disabling GroupNorm in ContextCNN.")

        # Block 1
        self.conv1 = DepthwiseSeparableConv(in_channels, out_features, kernel_size=3, padding=1, norm_groups=norm_groups)
        self.cbam1 = CBAM(out_features)

        # Block 2
        self.conv2 = DepthwiseSeparableConv(out_features, out_features, kernel_size=3, padding=1, norm_groups=norm_groups)
        # Note: CBAM might be added after conv2 as well if needed

        # Block 3 (Final conv block before residual, skip final ReLU within the block)
        use_norm3 = norm_groups > 0 and out_features >= norm_groups
        self.conv3_dw = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1, groups=out_features, bias=False)
        self.norm3_dw = nn.GroupNorm(norm_groups, out_features) if use_norm3 else nn.Identity()
        self.conv3_pw = nn.Conv2d(out_features, out_features, kernel_size=1, bias=False)
        self.norm3_pw = nn.GroupNorm(norm_groups, out_features) if use_norm3 else nn.Identity()
        # No final ReLU here in conv3 block

        self.cbam3 = CBAM(out_features)
        self.relu_final = nn.ReLU(inplace=True) # Final ReLU after residual addition

    def forward(self, x):
        x1 = self.conv1(x)
        x1_att = self.cbam1(x1) # Feature map after first block + attention

        x2 = self.conv2(x1_att) # Feature map after second block

        # Manual application for conv3 block (no internal ReLU)
        x3 = self.conv3_dw(x2)
        x3 = self.norm3_dw(x3)
        x3 = self.conv3_pw(x3)
        x3 = self.norm3_pw(x3) # Output just before attention and residual

        x3_att = self.cbam3(x3) # Apply attention to the output of conv3

        # Residual connection (using attention outputs)
        # Ensure shapes match for addition
        out = x1_att + x3_att
        return self.relu_final(out) # Apply final ReLU


# --- SIREN Activation & Initialization ---
class SineActivation(nn.Module):
    def __init__(self, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)

def sine_init(m, omega_0=30.0):
    """Initialize weights for non-first Sine activation layers."""
    if isinstance(m, nn.Linear):
        num_input = m.weight.size(-1)
        # SIREN paper supplement C.1: U ~ [-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0]
        bound = math.sqrt(6.0 / num_input) / omega_0
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def first_layer_sine_init(m):
    """Initialize weights for the first layer of a SIREN MLP."""
    if isinstance(m, nn.Linear):
        num_input = m.weight.size(-1)
        # SIREN paper C.1: U ~ [-1/n, 1/n]
        bound = 1.0 / num_input
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# --- Coordinate MLP with SIREN/FiLM Options ---
class CoordinateMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_depth, output_dim=4,
                 skip_connection_dim=0, skip_layer_index=None,
                 use_siren=False, siren_omega_0=30.0,
                 use_film=False, film_context_dim=0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.output_dim = output_dim
        self.skip_connection_dim = skip_connection_dim
        # Default skip index if not provided
        self.skip_layer_index = skip_layer_index if skip_layer_index is not None else hidden_depth // 2
        self.use_siren = use_siren
        self.siren_omega_0 = siren_omega_0
        self.use_film = use_film
        self.film_context_dim = film_context_dim

        if use_siren: print(f"MLP: Using SIREN activation (omega_0={siren_omega_0})")
        if use_film: print(f"MLP: Using FiLM modulation (context_dim={film_context_dim})")
        if skip_connection_dim > 0: print(f"MLP: Using skip connection (dim={skip_connection_dim}) at hidden layer {self.skip_layer_index}")

        layers = []
        film_layers = [] # To store FiLM generator layers if used
        current_dim = self.input_dim

        # Hidden layers
        for i in range(self.hidden_depth):
            # Check if skip connection should be injected *before* this layer's Linear
            linear_input_dim = current_dim
            if i == self.skip_layer_index and self.skip_connection_dim > 0:
                linear_input_dim += self.skip_connection_dim

            # Linear layer
            linear_layer = nn.Linear(linear_input_dim, self.hidden_dim)
            layers.append(linear_layer)

            # Normalization (LayerNorm is common in coordinate networks)
            # Apply *after* linear, *before* activation (and potentially FiLM)
            layers.append(nn.LayerNorm(self.hidden_dim))

            # FiLM Layer (applied after Norm, before Activation)
            if self.use_film and self.film_context_dim > 0:
                # Layer to generate gamma and beta from context
                film_gen = nn.Linear(self.film_context_dim, self.hidden_dim * 2) # *2 for gamma & beta
                film_layers.append(film_gen)
                # Note: FiLM application happens in the forward pass

            # Activation
            if self.use_siren:
                layers.append(SineActivation(omega_0=self.siren_omega_0 if i > 0 else self.siren_omega_0)) # Adjust omega for first layer? Often kept same.
            else:
                layers.append(nn.ReLU(inplace=True))

            # Update current_dim for the next layer's input
            current_dim = self.hidden_dim

        # Final output layer
        final_linear_input_dim = current_dim
        # Check for skip connection right before the final layer
        if self.hidden_depth == self.skip_layer_index and self.skip_connection_dim > 0:
            final_linear_input_dim += self.skip_connection_dim

        layers.append(nn.Linear(final_linear_input_dim, self.output_dim))

        self.mlp = nn.ModuleList(layers)
        self.film_layers = nn.ModuleList(film_layers) if self.use_film and film_layers else None

        # Apply SIREN initialization if used
        if self.use_siren:
            self.init_weights()

    def init_weights(self):
        """Applies SIREN specific weight initialization."""
        # Apply sine_init to all linear layers except the first
        self.mlp.apply(lambda m: sine_init(m, omega_0=self.siren_omega_0))
        # Apply first_layer_sine_init specifically to the first linear layer
        if self.mlp and isinstance(self.mlp[0], nn.Linear):
            first_layer_sine_init(self.mlp[0])
        print("Applied SIREN weight initialization to MLP.")


    def forward(self, x, skip_features=None, film_context=None):
        """
        Forward pass for the Coordinate MLP.

        Args:
            x (torch.Tensor): Input features (B, N, input_dim).
            skip_features (torch.Tensor, optional): Features for skip connection (B, N, skip_connection_dim).
            film_context (torch.Tensor, optional): Context for FiLM modulation (B, N, film_context_dim).

        Returns:
            torch.Tensor: Output features (B, N, output_dim).
        """
        hidden = x
        film_gen_idx = 0 # Index for film_layers list
        layer_idx = 0 # Index for the main mlp list
        current_hidden_layer_index = 0 # Tracks which logical hidden layer (0 to depth-1)

        while layer_idx < len(self.mlp):
            layer = self.mlp[layer_idx]

            # --- Apply Skip Connection (if applicable before Linear) ---
            # Check if current layer is Linear AND it's the target skip index
            if isinstance(layer, nn.Linear) and current_hidden_layer_index == self.skip_layer_index and \
               skip_features is not None and self.skip_connection_dim > 0:
                # Ensure skip features are on the same device
                skip_features = skip_features.to(hidden.device)
                # Concatenate along the feature dimension
                hidden = torch.cat([hidden, skip_features], dim=-1)

            # --- Apply the current layer (Linear, Norm, or Activation) ---
            hidden = layer(hidden)
            layer_idx += 1

            # --- Apply FiLM (if applicable after Norm) ---
            # Check if the *previous* layer was Norm AND FiLM is enabled
            if isinstance(layer, nn.LayerNorm) and self.use_film and self.film_layers is not None and \
               film_gen_idx < len(self.film_layers):

                if film_context is None:
                    raise ValueError("FiLM context required but not provided.")

                film_gen_layer = self.film_layers[film_gen_idx]
                film_context = film_context.to(hidden.device) # Ensure device match

                # Generate gamma and beta
                # film_context might be (B, N, Ctx) or just (B, Ctx) - ensure it's compatible
                # Assuming film_context is (B, N, film_context_dim) to match hidden (B, N, hidden_dim)
                gamma_beta = film_gen_layer(film_context) # Output: (B, N, hidden_dim * 2)

                # Split into gamma and beta
                gamma, beta = torch.chunk(gamma_beta, 2, dim=-1) # Each: (B, N, hidden_dim)

                # Apply FiLM: y = gamma * x + beta
                hidden = gamma * hidden + beta
                film_gen_idx += 1

            # --- Increment hidden layer counter after Activation ---
            if isinstance(layer, (nn.ReLU, SineActivation)):
                current_hidden_layer_index += 1

        # The final layer (output) is already applied in the loop
        return hidden


# --- Main RGB2RAW Coordinate Network Model ---
class RGB2RAWCoordNet(nn.Module):
    def __init__(self, mlp_width=128, mlp_depth=5, context_cnn_features=16,
                 global_context_dim=16, pos_encoding_levels=10, include_sampled_rgb=True,
                 mlp_skip_layer_index=None, config=None): # Pass full config dict
        super().__init__()

        if config is None:
            raise ValueError("Config dictionary must be provided to RGB2RAWCoordNet")
        self.config = config
        self.include_sampled_rgb = include_sampled_rgb

        # --- Context CNN ---
        # Use Depthwise Separable CNN based on config flag
        if config.get("use_depthwise_separable_conv", False):
            self.context_cnn = ContextCNN(
                in_channels=3,
                out_features=context_cnn_features,
                norm_groups=config.get("context_cnn_norm_groups", 4) # Default to 4 groups
            )
        else:
            # If not using depthwise, you'd need a definition for a standard ContextCNN here
            # For example:
            # self.context_cnn = StandardContextCNN(...)
            raise NotImplementedError("Standard ContextCNN (non-depthwise) not defined in model.py. Set 'use_depthwise_separable_conv' to True in config or implement standard CNN.")

        # --- Global Context Path ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_processor = nn.Linear(context_cnn_features, global_context_dim)
        self.global_context_dim = global_context_dim

        # --- Positional Encoding for Coordinates ---
        self.pos_encoder = PositionalEncoding(
            num_encoding_functions=pos_encoding_levels,
            include_input=True # Usually include raw coordinates
        )
        # Get PE output dimension based on 2D coordinates
        self.pos_enc_dim = self.pos_encoder.get_output_dim(input_dims=2)

        # --- Feature Dimensions ---
        self.context_dim = context_cnn_features
        self.rgb_dim = 3 if self.include_sampled_rgb else 0

        # --- Skip Connection Dimension (using Positional Encoding) ---
        self.mlp_skip_connection_dim = self.pos_enc_dim
        self.mlp_skip_layer_index = mlp_skip_layer_index # Passed from config or default

        # --- Calculate MLP Input Dimension ---
        mlp_input_dim = self.pos_enc_dim + self.context_dim + self.rgb_dim + self.global_context_dim
        print(f"MLP Initial Input Dim Calculation:")
        print(f"  Positional Encoding (coords): {self.pos_enc_dim}")
        print(f"  Sampled CNN Context:          {self.context_dim}")
        print(f"  Sampled RGB:                  {self.rgb_dim}")
        print(f"  Global Context:               {self.global_context_dim}")
        print(f"  -----------------------------------")
        print(f"  Total Initial MLP Input Dim:  {mlp_input_dim}")
        if self.mlp_skip_connection_dim > 0:
             print(f"MLP Skip Connection (PosEnc): Dim={self.mlp_skip_connection_dim} at Layer Index={self.mlp_skip_layer_index if self.mlp_skip_layer_index is not None else mlp_depth // 2}")

        # --- Determine FiLM Context Dimension for MLP ---
        self.film_context_dim_mlp = 0
        if config.get("use_film", False):
            film_source = config.get("film_context_source", "global") # Default to global
            if film_source == "global":
                self.film_context_dim_mlp = self.global_context_dim
            elif film_source == "local":
                self.film_context_dim_mlp = self.context_dim # Use sampled local features
            else:
                print(f"Warning: Unknown film_context_source '{film_source}', defaulting to 'global'.")
                self.film_context_dim_mlp = self.global_context_dim
            print(f"MLP FiLM Context: Source='{film_source}', Dim={self.film_context_dim_mlp}")

        # --- Coordinate MLP ---
        self.mlp = CoordinateMLP(
            input_dim=mlp_input_dim,
            hidden_dim=mlp_width,
            hidden_depth=mlp_depth,
            output_dim=4, # Output: RGGB RAW channels
            skip_connection_dim=self.mlp_skip_connection_dim,
            skip_layer_index=self.mlp_skip_layer_index,
            use_siren=config.get("use_siren", False),
            siren_omega_0=config.get("siren_omega_0", 30.0),
            use_film=config.get("use_film", False),
            film_context_dim=self.film_context_dim_mlp
        )


    def forward(self, rgb_img, coords_raw_norm):
        """
        Forward pass for predicting RAW values at given coordinates.

        Args:
            rgb_img (torch.Tensor): Input RGB image tensor (B, 3, H_rgb, W_rgb).
            coords_raw_norm (torch.Tensor): Normalized RAW coordinates (B, N_samples, 2) in range [-1, 1].
                                             N_samples is the number of points to predict per image.

        Returns:
            torch.Tensor: Predicted RAW values (B, N_samples, 4) for RGGB channels.
        """
        B, _, H_rgb, W_rgb = rgb_img.shape
        if H_rgb % 2 != 0 or W_rgb % 2 != 0:
             raise ValueError(f"Input RGB dimensions must be even, but got {H_rgb}x{W_rgb}")
        H_raw_eff, W_raw_eff = H_rgb // 2, W_rgb // 2 # Effective RAW dimensions

        # 1. Extract Context Features using CNN
        context_features = self.context_cnn(rgb_img) # (B, C_feat, H_rgb, W_rgb)

        # 2. Compute Global Context
        global_features_pooled = self.global_pool(context_features).squeeze(-1).squeeze(-1) # (B, C_feat)
        global_context = self.global_processor(global_features_pooled) # (B, global_context_dim)

        # 3. Positional Encoding for Query Coordinates
        coords_raw_norm = coords_raw_norm.to(rgb_img.device)
        pos_enc = self.pos_encoder(coords_raw_norm) # (B, N_samples, pos_enc_dim)
        N_samples = pos_enc.shape[1] # Number of queried points

        # 4. Sample Local Context Features using grid_sample
        # Map normalized RAW coords [-1, 1] to normalized RGB coords [-1, 1] for grid_sample
        # coords_raw_norm (x,y) -> pixel coords (raw) -> pixel coords (rgb center) -> normalized coords (rgb)
        coords_raw_pix = (coords_raw_norm + 1.0) * 0.5 # Map to [0, 1] range
        # Scale to pixel coordinates in the effective RAW grid
        coords_raw_pix_scaled_x = coords_raw_pix[..., 0] * (W_raw_eff - 1)
        coords_raw_pix_scaled_y = coords_raw_pix[..., 1] * (H_raw_eff - 1)
        # Map to center pixel coordinates in the RGB grid
        # Center of 2x2 RAW block corresponds roughly to RGB pixel center
        coords_rgb_pix_center_x = coords_raw_pix_scaled_x * 2.0 + 0.5
        coords_rgb_pix_center_y = coords_raw_pix_scaled_y * 2.0 + 0.5
        # Normalize RGB pixel coordinates back to [-1, 1] for grid_sample
        # grid_sample expects (x, y) order
        coords_rgb_norm_x = (coords_rgb_pix_center_x / max(1, W_rgb - 1)) * 2.0 - 1.0
        coords_rgb_norm_y = (coords_rgb_pix_center_y / max(1, H_rgb - 1)) * 2.0 - 1.0
        # Stack into (B, N_samples, 2) tensor for grid_sample
        grid = torch.stack([coords_rgb_norm_x, coords_rgb_norm_y], dim=-1).unsqueeze(2) # (B, N_samples, 1, 2)

        # Sample from context_features (B, C_feat, H_rgb, W_rgb)
        sampled_context = F.grid_sample(
            context_features, grid,
            mode='bilinear', padding_mode='border', align_corners=False # align_corners=False common for pixel centers
        ).squeeze(-1).permute(0, 2, 1) # -> (B, C_feat, N_samples) -> (B, N_samples, C_feat)

        # 5. Sample RGB values at query locations (optional)
        sampled_rgb = None
        if self.include_sampled_rgb:
            sampled_rgb = F.grid_sample(
                rgb_img, grid, # Use the same grid as context sampling
                mode='bilinear', padding_mode='border', align_corners=False
            ).squeeze(-1).permute(0, 2, 1) # -> (B, 3, N_samples) -> (B, N_samples, 3)

        # 6. Expand Global Context for each sample point
        # global_context shape: (B, global_context_dim)
        global_context_expanded = global_context.unsqueeze(1).expand(-1, N_samples, -1) # (B, N_samples, global_context_dim)

        # 7. Concatenate all features for MLP input
        mlp_inputs = [pos_enc, sampled_context, global_context_expanded]
        if self.include_sampled_rgb and sampled_rgb is not None:
            mlp_inputs.append(sampled_rgb)
        mlp_input_combined = torch.cat(mlp_inputs, dim=-1) # (B, N_samples, mlp_input_dim)

        # 8. Prepare FiLM Context (if used)
        film_context_input = None
        if self.config.get("use_film", False):
            film_source = self.config.get("film_context_source", "global")
            if film_source == "global":
                film_context_input = global_context_expanded # Shape (B, N_samples, global_dim)
            elif film_source == "local":
                film_context_input = sampled_context # Shape (B, N_samples, context_dim)
            # Ensure film_context_input has the correct dimension (film_context_dim_mlp)
            # If dimensions mismatch (e.g., local context != film dim), projection might be needed
            # Assuming direct use is intended based on config setup

        # 9. Pass through MLP
        raw_pred = self.mlp(
            mlp_input_combined,
            skip_features=pos_enc if self.mlp_skip_connection_dim > 0 else None, # Pass PosEnc as skip feature
            film_context=film_context_input # Pass prepared FiLM context
        ) # Output: (B, N_samples, 4)

        # 10. Apply final activation (e.g., Sigmoid) if needed to constrain output range [0, 1]
        # Often handled implicitly by loss or done post-prediction if necessary.
        # Let's assume loss function handles range, or add sigmoid if needed:
        # raw_pred = torch.sigmoid(raw_pred)

        return raw_pred


    @torch.no_grad()
    def predict_full_image(self, rgb_img_batch, chunk_size=-1):
        """
        Predicts the full RAW image(s) corresponding to the input RGB batch.
        Handles padding and chunking for memory efficiency.

        Args:
            rgb_img_batch (torch.Tensor): Batch of input RGB images (B, 3, H, W).
                                         Must have even dimensions (padding applied if needed).
            chunk_size (int): Number of pixels to process in one MLP forward pass.
                              -1 means process all pixels at once.

        Returns:
            torch.Tensor: Predicted full RAW image batch (B, 4, H_raw, W_raw).
        """
        self.eval() # Ensure model is in evaluation mode
        B, _, H_rgb, W_rgb = rgb_img_batch.shape
        device = rgb_img_batch.device

        # Ensure input dimensions are even (required by the coordinate mapping)
        pad_h = (2 - H_rgb % 2) % 2
        pad_w = (2 - W_rgb % 2) % 2
        if pad_h > 0 or pad_w > 0:
            # Use reflection padding to avoid border artifacts
            rgb_img_batch = F.pad(rgb_img_batch, (0, pad_w, 0, pad_h), mode='reflect')
            B, _, H_rgb, W_rgb = rgb_img_batch.shape # Update dimensions

        H_raw, W_raw = H_rgb // 2, W_rgb // 2
        if H_raw <= 0 or W_raw <= 0:
            print(f"Warning: Input image size {H_rgb}x{W_rgb} results in non-positive RAW dimensions {H_raw}x{W_raw}. Returning zeros.")
            # Return zero tensor with expected channel dim but minimal spatial dim
            return torch.zeros(B, 4, max(1, H_raw), max(1, W_raw), device=device)

        # Create a grid of normalized coordinates for the full RAW image
        y_coords = torch.linspace(-1.0, 1.0, H_raw, device=device)
        x_coords = torch.linspace(-1.0, 1.0, W_raw, device=device)
        # Create grid (H_raw, W_raw, 2) with (x, y) coordinates
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij') # indexing='ij' -> H, W output shapes
        # Stack and flatten: (H_raw * W_raw, 2)
        coords_raw_norm_flat = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        num_pixels = coords_raw_norm_flat.shape[0]

        # Determine effective chunk size
        if chunk_size is None or chunk_size <= 0:
             effective_chunk_size = num_pixels
        else:
             effective_chunk_size = min(chunk_size, num_pixels)

        all_preds = []
        # Process in chunks
        for i in range(0, num_pixels, effective_chunk_size):
            # Get coordinate chunk (N_chunk, 2)
            coord_chunk = coords_raw_norm_flat[i : i + effective_chunk_size]
            # Add batch dimension and expand: (B, N_chunk, 2)
            coord_chunk_batched = coord_chunk.unsqueeze(0).expand(B, -1, -1)

            # Run forward pass for the chunk
            # The forward pass expects (B, 3, H_rgb, W_rgb) and (B, N_chunk, 2)
            pred_chunk = self.forward(rgb_img_batch, coord_chunk_batched) # Output: (B, N_chunk, 4)
            all_preds.append(pred_chunk.cpu()) # Move to CPU to save GPU memory

        # Concatenate predictions from all chunks along the 'N' dimension
        full_raw_pred_batched = torch.cat(all_preds, dim=1) # (B, num_pixels, 4)

        # Reshape back to image format (B, 4, H_raw, W_raw)
        full_raw_pred = full_raw_pred_batched.permute(0, 2, 1).reshape(B, 4, H_raw, W_raw)

        # Remove padding if it was added
        if pad_h > 0 or pad_w > 0:
            orig_H_raw = (H_rgb - pad_h) // 2
            orig_W_raw = (W_rgb - pad_w) // 2
            # Crop the prediction back to the original RAW size
            full_raw_pred = full_raw_pred[:, :, :orig_H_raw, :orig_W_raw]

        return full_raw_pred.to(device) # Return on the original device
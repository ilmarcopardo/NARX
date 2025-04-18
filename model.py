import torch
import torch.nn as nn
import inspect

class NARX(nn.Module):
    def __init__(self, d_i: int, d_o: int, d_x: int, d_y: int, d_hl: int, act_func: str = "Sigmoid"):
        super().__init__()

        if not isinstance(d_i, int) or d_i <= 0:
            raise ValueError(f"Input delay d_i must be a positive integer, got {d_i}")
        if not isinstance(d_o, int) or d_o <= 0:
            raise ValueError(f"Output delay d_o must be a positive integer, got {d_o}")
        if not isinstance(d_x, int) or d_x <= 0:
            raise ValueError(f"Input dimension d_x must be a positive integer, got {d_x}")
        if not isinstance(d_y, int) or d_y <= 0:
            raise ValueError(f"Output dimension d_y must be a positive integer, got {d_y}")
        if not isinstance(d_hl, int) or d_hl <= 0:
            raise ValueError(f"Hidden layer dimension d_hl must be a positive integer, got {d_hl}")
        if not isinstance(act_func, str):
             raise TypeError(f"act_func must be a string, got {type(act_func)}")

        self.d_i = d_i
        self.d_o = d_o
        self.d_x = d_x
        self.d_y = d_y
        self.d_hl = d_hl
        self.act_func = act_func

        self.input_size = self.d_i * self.d_x + self.d_o * self.d_y
        self.hl1 = nn.Linear(self.input_size, self.d_hl)
        self.hl2 = nn.Linear(self.d_hl, self.d_y)

        self.act1 = nn.Tanh()

        try:
            activation_class = getattr(torch.nn, self.act_func)
            if inspect.isclass(activation_class) and issubclass(activation_class, torch.nn.Module):
                self.act2 = activation_class()
            else:
                raise TypeError(f"'{self.act_func}' found in torch.nn, but it is not an nn.Module subclass.")
        except AttributeError:
            raise ValueError(f"Activation function '{self.act_func}' not found in torch.nn module.")
        except TypeError as e:
             raise TypeError(f"Error processing activation '{self.act_func}': {e}")


    def forward(self, x: torch.Tensor, mode: str = "close", y: torch.Tensor = None, bootstrap: int = None) -> torch.Tensor:

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input x must be a torch.Tensor, got {type(x)}")
        if x.ndim != 3:
            raise ValueError(f"Input x must be 3-dimensional (batch, steps, features), got {x.ndim} dimensions")

        batch_size, num_steps, input_dim = x.shape

        if input_dim != self.d_x:
             raise ValueError(f"Input dimension mismatch. Expected {self.d_x}, got {input_dim}")

        required_past_steps = max(self.d_i, self.d_o)
        if num_steps < required_past_steps:
            raise ValueError(f"Time series length {num_steps} is too short. Need at least {required_past_steps} steps based on d_i={self.d_i} and d_o={self.d_o}")

        if mode != "close" and mode != "open":
            raise ValueError(f"A valid mode must be selected, got {mode}.")

        if y is not None:
            if not isinstance(y, torch.Tensor):
                 raise TypeError(f"Input y must be a torch.Tensor when provided, got {type(y)}")
            if y.ndim != 3:
                 raise ValueError(f"Input y must be 3-dimensional (batch, steps, features), got {y.ndim} dimensions")
            if y.shape[0] != batch_size or y.shape[1] != num_steps:
                 raise ValueError(f"Shape mismatch between x {x.shape} and y {y.shape}. Batch size and num_steps must match.")
            if y.shape[2] != self.d_y:
                 raise ValueError(f"Output dimension mismatch. Expected {self.d_y} from d_y, got {y.shape[2]} in y tensor.")
            
        if bootstrap is not None:
            if y is None:
                raise ValueError("If bootstrap is specified, y must also be provided.")
            if not isinstance(bootstrap, int) or bootstrap <= 0:
                 raise ValueError(f"Bootstrap steps must be a positive integer, got {bootstrap}")
            if bootstrap > num_steps:
                 raise ValueError(f"Bootstrap steps {bootstrap} cannot be greater than sequence length {num_steps}")
            if bootstrap < required_past_steps:
                 raise ValueError(f"Bootstrap steps {bootstrap} must be at least {required_past_steps} (max(d_i, d_o)) to make the first prediction.")

        y_pred = torch.zeros(batch_size, num_steps, self.d_y, device=x.device, dtype=x.dtype)

        if bootstrap is not None:
            copy_len = min(bootstrap, y.shape[1])
            y_pred[:, :copy_len, :] = y[:, :copy_len, :]

        loop_start_idx = bootstrap if bootstrap is not None else required_past_steps

        for t in range(loop_start_idx, num_steps):
            x_window = x[:, t - self.d_i : t, :].reshape(batch_size, -1)

            if y is not None and bootstrap is not None and t < bootstrap:
                 y_window_source = y
            else:
                 y_window_source = y_pred # Use self-generated predictions

            y_window = y_window_source[:, t - self.d_o : t, :].reshape(batch_size, -1)

            input_cat = torch.cat((x_window, y_window), dim=1)

            hidden = self.act1(self.hl1(input_cat))
            output = self.act2(self.hl2(hidden))

            y_pred[:, t, :] = output

        return y_pred

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_dim = 4
    x_length = 20
    d_x = 3
    d_y = 2
    d_i = 5
    d_o = 4
    d_hl = 16

    model = NARX(d_i=d_i, d_o=d_o, d_x=d_x, d_y=d_y, d_hl=d_hl).to(device)

    x = torch.rand(batch_dim, x_length, d_x, device=device)
    y_true = torch.rand(batch_dim, x_length, d_y, device=device) 

    # Mode 1: Parallel (free-running)
    print("Testing Parallel Mode...")
    y_pred_parallel = model(x)
    assert y_pred_parallel.shape == (batch_dim, x_length, d_y), \
        f"Parallel Mode: Unexpected shape: {y_pred_parallel.shape}, expected ({batch_dim}, {x_length}, {d_y})"
    print("Parallel Mode OK.")

    # Mode 2: Series-Parallel (teacher forcing)
    print("\nTesting Series-Parallel Mode...")
    y_pred_series_parallel = model(x, y=y_true)
    assert y_pred_series_parallel.shape == (batch_dim, x_length, d_y), \
        f"Series-Parallel Mode: Unexpected shape: {y_pred_series_parallel.shape}, expected ({batch_dim}, {x_length}, {d_y})"
    print("Series-Parallel Mode OK.")

    # Mode 3: Bootstrap
    print("\nTesting Bootstrap Mode...")
    bootstrap_steps = max(d_i, d_o) + 2 # Example: Ensure enough history + a few steps
    if bootstrap_steps > x_length:
        bootstrap_steps = x_length # Cannot bootstrap more than available steps
        print(f"Warning: Bootstrap steps reduced to {bootstrap_steps} due to sequence length.")

    if x_length >= max(d_i, d_o):
        y_pred_bootstrap = model(x, y=y_true, bootstrap=bootstrap_steps)
        assert y_pred_bootstrap.shape == (batch_dim, x_length, d_y), \
            f"Bootstrap Mode: Unexpected shape: {y_pred_bootstrap.shape}, expected ({batch_dim}, {x_length}, {d_y})"

        # Check if initial steps match y_true
        assert torch.allclose(y_pred_bootstrap[:, :bootstrap_steps, :], y_true[:, :bootstrap_steps, :]), \
            "Bootstrap Mode: Initial steps do not match y_true"

        # Check if later steps are different (unless y_true happens to be the perfect prediction)
        # We expect predictions after bootstrap to diverge usually
        if bootstrap_steps < x_length:
             # Check a step right after bootstrap ends
             assert not torch.allclose(y_pred_bootstrap[:, bootstrap_steps, :], y_true[:, bootstrap_steps, :]), \
                 f"Bootstrap Mode: Prediction at step {bootstrap_steps} unexpectedly matched y_true (highly unlikely unless model is perfect or lucky)"
        print("Bootstrap Mode OK.")
    else:
        print("Skipping Bootstrap test due to insufficient sequence length.")

    # Test Error Handling
    print("\nTesting Error Handling...")
    try:
        short_x = torch.rand(batch_dim, d_i - 1, d_x, device=device)
        model(short_x)
    except ValueError as e:
        print(f"Caught expected error for short sequence: {e}")

    try:
        model(x, bootstrap=5) # Missing y
    except ValueError as e:
        print(f"Caught expected error for bootstrap without y: {e}")

    try:
        wrong_dim_x = torch.rand(batch_dim, x_length, d_x + 1, device=device)
        model(wrong_dim_x)
    except ValueError as e:
        print(f"Caught expected error for wrong x dimension: {e}")

    try:
        wrong_dim_y = torch.rand(batch_dim, x_length, d_y + 1, device=device)
        model(x, y=wrong_dim_y)
    except ValueError as e:
        print(f"Caught expected error for wrong y dimension: {e}")

    try:
        wrong_shape_y = torch.rand(batch_dim, x_length -1, d_y, device=device)
        model(x, y=wrong_shape_y)
    except ValueError as e:
        print(f"Caught expected error for y shape mismatch: {e}")

    try:
        model(x, y=y_true, bootstrap=max(d_i, d_o)-1) # Bootstrap too short
    except ValueError as e:
        print(f"Caught expected error for insufficient bootstrap steps: {e}")

    print("Error handling tests completed.")
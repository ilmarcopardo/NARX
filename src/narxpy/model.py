import torch
import torch.nn as nn
import inspect

class NARX(nn.Module):
    def __init__(self, d_i: int, d_o: int, d_x: int, d_y: int, d_hl: int, act_func: str = "Sigmoid"):
        """
        Initializes the NARX (Nonlinear Autoregressive Network with Exogenous Inputs) model.

        Args:
            d_i (int): Input delay (lookback window size for exogenous inputs x).
            d_o (int): Output delay (lookback window size for autoregressive feedback y).
            d_x (int): Dimension of the exogenous input features at each time step.
            d_y (int): Dimension of the output features at each time step.
            d_hl (int): Dimension of the hidden layer.
            act_func (str, optional): Name of the torch.nn activation function to use
                for the output layer. Defaults to "Sigmoid".

        Raises:
            ValueError: If any dimension (d_i, d_o, d_x, d_y, d_hl) is not a positive integer.
            TypeError: If act_func is not a string.
            ValueError: If act_func name is not found in torch.nn.
            TypeError: If act_func name corresponds to something in torch.nn that is not
                       an nn.Module subclass.
        """

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
        """
        Performs the forward pass for the NARX model in open or closed loop.

        Generates predictions step-by-step through the time series. The source
        of the autoregressive feedback (past outputs) depends on the specified mode.

        Note: The terminology "open" and "close" loop used here aligns with
        common usage in MATLAB's NARX simulation, where:
          - "open": Uses provided true target values `y` for feedback (Series-Parallel / Teacher Forcing).
          - "close": Uses the model's own past predictions for feedback (Parallel / Simulation).

        Args:
            x (torch.Tensor): The input time series tensor with shape
                (batch_size, num_steps, d_x).
            mode (str, optional): Specifies the feedback loop mode.
                - "close": Uses the model's own previous predictions (`y_pred`) as
                  feedback (Parallel architecture). Default is "close".
                - "open": Uses the provided true sequence `y` as feedback
                  (Series-Parallel architecture / Teacher Forcing).
                  Requires `y` to be provided.
            y (torch.Tensor, optional): The target (true output) time series tensor
                with shape (batch_size, num_steps, d_y). Required if `mode` is "open"
                or if `bootstrap` is used. Defaults to None.
            bootstrap (int, optional): If provided (and `mode` is "close"), uses the
                first `bootstrap` steps from the provided `y` tensor to initialize
                the `y_pred` tensor. The prediction loop will then start from the
                `bootstrap`-th step. Requires `y` to be provided. Cannot be used
                if `mode` is "open". Defaults to None.

        Returns:
            torch.Tensor: The predicted output time series tensor with shape
                (batch_size, num_steps, d_y).

        Raises:
            TypeError: If `x` or `y` (if provided) are not torch.Tensors.
            ValueError: If `x` or `y` (if provided) do not have 3 dimensions.
            ValueError: If the input dimension of `x` does not match `self.d_x`.
            ValueError: If `num_steps` in `x` is less than `max(self.d_i, self.d_o)`.
            ValueError: If an invalid `mode` string is provided (not "close" or "open").
            ValueError: If `mode` is "open" but `y` is not provided.
            ValueError: If `mode` is "open" and `bootstrap` is also provided (not allowed).
            ValueError: If shape of `y` does not match `x` in batch_size and num_steps,
                        or if the feature dimension of `y` does not match `self.d_y`.
            ValueError: If `bootstrap` is provided but `y` is None.
            ValueError: If `bootstrap` is not a positive integer.
            ValueError: If `bootstrap` is greater than `num_steps`.
            ValueError: If `bootstrap` is less than `max(self.d_i, self.d_o)`.
        """

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
        if mode == "open" and y is None:
            raise ValueError(f"Selected mode is {mode} loop but no input y was given")
        if mode == "open" and bootstrap is not None:
            raise ValueError(f"Bootstrap is not needed if mode is {mode} loop.")

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

            if mode == "open": y_window_source = y
            else: y_window_source = y_pred 

            y_window = y_window_source[:, t - self.d_o : t, :].reshape(batch_size, -1)

            input_cat = torch.cat((x_window, y_window), dim=1)

            hidden = self.act1(self.hl1(input_cat))
            output = self.act2(self.hl2(hidden))

            y_pred[:, t, :] = output

        return y_pred

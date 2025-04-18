import pytest
import torch
import torch.nn as nn

from . import NARX  

# --- Fixtures for Reusable Test Setup ---

@pytest.fixture(scope="module") # Reuse parameters for all tests in this module
def model_params():
    """Provides standard model parameters."""
    return {
        "d_i": 5,
        "d_o": 4,
        "d_x": 3,
        "d_y": 2,
        "d_hl": 16,
        "act_func": "Tanh" 
    }

@pytest.fixture
def narx_model(model_params):
    """Provides a default NARX model instance."""
    return NARX(**model_params)

@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def num_steps(model_params):
    # Ensure enough steps for testing
    return max(model_params['d_i'], model_params['d_o']) + 10

@pytest.fixture
def valid_x(batch_size, num_steps, model_params):
    """Provides a valid input tensor x."""
    # Use values slightly away from zero for Tanh activation testing
    return torch.randn(batch_size, num_steps, model_params['d_x']) * 0.5 + 0.1

@pytest.fixture
def valid_y(batch_size, num_steps, model_params):
    """Provides a valid target tensor y."""
    return torch.randn(batch_size, num_steps, model_params['d_y']) * 0.5 + 0.1


# --- Initialization Tests ---

def test_init_success(narx_model, model_params):
    """Tests successful initialization with valid parameters."""
    assert narx_model.d_i == model_params['d_i']
    assert narx_model.d_o == model_params['d_o']
    assert narx_model.d_x == model_params['d_x']
    assert narx_model.d_y == model_params['d_y']
    assert narx_model.d_hl == model_params['d_hl']
    assert isinstance(narx_model.hl1, nn.Linear)
    assert isinstance(narx_model.hl2, nn.Linear)
    assert isinstance(narx_model.act1, nn.Tanh)
    assert isinstance(narx_model.act2, getattr(nn, model_params['act_func']))

    expected_input_size = model_params['d_i'] * model_params['d_x'] + \
                          model_params['d_o'] * model_params['d_y']
    assert narx_model.hl1.in_features == expected_input_size
    assert narx_model.hl1.out_features == model_params['d_hl']
    assert narx_model.hl2.in_features == model_params['d_hl']
    assert narx_model.hl2.out_features == model_params['d_y']

def test_init_default_activation():
    """Tests initialization uses default activation if not specified."""
    model = NARX(d_i=3, d_o=3, d_x=2, d_y=1, d_hl=8) 
    assert isinstance(model.act2, nn.Sigmoid)

@pytest.mark.parametrize("invalid_dim", [0, -1, 1.5])
@pytest.mark.parametrize("dim_name", ["d_i", "d_o", "d_x", "d_y", "d_hl"])
def test_init_invalid_dims(model_params, invalid_dim, dim_name):
    """Tests initialization fails with non-positive integer dimensions."""
    params = model_params.copy()
    params[dim_name] = invalid_dim
    with pytest.raises(ValueError, match=f"{dim_name} must be a positive integer"):
        NARX(**params)

def test_init_invalid_act_func_type(model_params):
    """Tests initialization fails with non-string activation function."""
    params = model_params.copy()
    params["act_func"] = nn.ReLU 
    with pytest.raises(TypeError, match="act_func must be a string"):
        NARX(**params)

def test_init_invalid_act_func_name(model_params):
    """Tests initialization fails with an unknown activation function name."""
    params = model_params.copy()
    params["act_func"] = "NonExistentActivation"
    with pytest.raises(ValueError, match="Activation function .* not found"):
        NARX(**params)

def test_init_invalid_act_func_target(model_params):
    """Tests initialization fails if name points to non-Module in torch.nn."""
    params = model_params.copy()
    params["act_func"] = "functional" # Exists but is not an nn.Module subclass
    with pytest.raises(TypeError, match=".* but it is not an nn.Module subclass"):
        NARX(**params)

# --- Forward Pass Validation Tests ---

def test_forward_invalid_x_type(narx_model):
    """Tests forward pass fails with non-Tensor input."""
    with pytest.raises(TypeError, match="Input x must be a torch.Tensor"):
        narx_model([1, 2, 3]) # Pass a list

def test_forward_invalid_x_dims(narx_model, valid_x):
    """Tests forward pass fails with incorrect number of dimensions for x."""
    with pytest.raises(ValueError, match="Input x must be 3-dimensional"):
        narx_model(valid_x[0]) # Pass only one batch item (2D)
    with pytest.raises(ValueError, match="Input x must be 3-dimensional"):
        narx_model(valid_x.unsqueeze(0)) # Pass 4D tensor

def test_forward_incorrect_x_features(narx_model, valid_x):
    """Tests forward pass fails if x feature dimension doesn't match d_x."""
    x_wrong_feat = torch.randn_like(valid_x[:, :, :-1]) # Remove last feature
    with pytest.raises(ValueError, match="Input dimension mismatch"):
        narx_model(x_wrong_feat)

def test_forward_insufficient_seq_len(narx_model, model_params, batch_size):
    """Tests forward pass fails if sequence length is too short."""
    required_steps = max(model_params['d_i'], model_params['d_o'])
    x_short = torch.randn(batch_size, required_steps - 1, model_params['d_x'])
    with pytest.raises(ValueError, match="Time series length .* is too short"):
        narx_model(x_short)

def test_forward_invalid_mode(narx_model, valid_x):
    """Tests forward pass fails with invalid mode string."""
    with pytest.raises(ValueError, match="A valid mode must be selected"):
        narx_model(valid_x, mode="invalid_mode")

def test_forward_open_mode_no_y(narx_model, valid_x):
    """Tests forward pass fails if mode is 'open' but y is not provided."""
    with pytest.raises(ValueError, match="Selected mode is open loop but no input y was given"):
        narx_model(valid_x, mode="open", y=None)

def test_forward_open_mode_with_bootstrap(narx_model, valid_x, valid_y):
    """Tests forward pass fails if mode is 'open' and bootstrap is provided."""
    with pytest.raises(ValueError, match="Bootstrap is not needed if mode is open loop"):
        narx_model(valid_x, mode="open", y=valid_y, bootstrap=5)

@pytest.mark.parametrize("y_input", [None, [1,2,3]]) # Test None covered by other tests
def test_forward_invalid_y_type(narx_model, valid_x, y_input):
    """Tests forward pass fails with non-Tensor y (when y is expected)."""
    # Need mode="open" or bootstrap to force validation of y
    if y_input is None: # Skip None test here, covered elsewhere
         pytest.skip("None y tested in mode='open' test")
    with pytest.raises(TypeError, match="Input y must be a torch.Tensor"):
         narx_model(valid_x, mode="close", y=y_input, bootstrap=valid_x.shape[1]-1)

def test_forward_invalid_y_dims(narx_model, valid_x, valid_y):
    """Tests forward pass fails with incorrect number of dimensions for y."""
    with pytest.raises(ValueError, match="Input y must be 3-dimensional"):
        narx_model(valid_x, mode="open", y=valid_y[0]) # Pass 2D y

def test_forward_y_shape_mismatch(narx_model, valid_x, valid_y):
    """Tests forward pass fails if y shape doesn't match x."""
    y_wrong_batch = valid_y[:-1, :, :]
    y_wrong_steps = valid_y[:, :-1, :]
    with pytest.raises(ValueError, match="Shape mismatch between x .* and y"):
        narx_model(valid_x, mode="open", y=y_wrong_batch)
    with pytest.raises(ValueError, match="Shape mismatch between x .* and y"):
        narx_model(valid_x, mode="open", y=y_wrong_steps)

def test_forward_incorrect_y_features(narx_model, valid_x, valid_y):
    """Tests forward pass fails if y feature dimension doesn't match d_y."""
    y_wrong_feat = torch.randn_like(valid_y[:, :, :-1]) # Remove last feature
    with pytest.raises(ValueError, match="Output dimension mismatch.* in y"):
        narx_model(valid_x, mode="open", y=y_wrong_feat)

def test_forward_bootstrap_no_y(narx_model, valid_x):
    """Tests forward pass fails if bootstrap is given but y is None."""
    with pytest.raises(ValueError, match="If bootstrap is specified, y must also be provided"):
        narx_model(valid_x, mode="close", y=None, bootstrap=5)

@pytest.mark.parametrize("bs_val", [0, -1, 5.5])
def test_forward_invalid_bootstrap_value(narx_model, valid_x, valid_y, bs_val):
    """Tests forward pass fails with non-positive integer bootstrap."""
    with pytest.raises((ValueError, TypeError), match="Bootstrap steps must be a positive integer"):
        narx_model(valid_x, mode="close", y=valid_y, bootstrap=bs_val)

def test_forward_bootstrap_too_large(narx_model, valid_x, valid_y, num_steps):
    """Tests forward pass fails if bootstrap > num_steps."""
    with pytest.raises(ValueError, match="Bootstrap steps .* cannot be greater than sequence length"):
        narx_model(valid_x, mode="close", y=valid_y, bootstrap=num_steps + 1)

def test_forward_bootstrap_too_small(narx_model, valid_x, valid_y, model_params):
    """Tests forward pass fails if bootstrap is less than required history."""
    required_steps = max(model_params['d_i'], model_params['d_o'])
    if required_steps > 1: # Only test if requirement is > 1
        with pytest.raises(ValueError, match="Bootstrap steps .* must be at least"):
             narx_model(valid_x, mode="close", y=valid_y, bootstrap=required_steps - 1)
    else:
        pytest.skip("Cannot test bootstrap too small if required steps is 1.")


# --- Forward Pass Functionality Tests ---

def test_forward_close_mode(narx_model, valid_x, model_params, batch_size, num_steps):
    """Tests forward pass in default 'close' mode."""
    y_pred = narx_model(valid_x, mode="close") # Default mode
    assert isinstance(y_pred, torch.Tensor)
    assert y_pred.shape == (batch_size, num_steps, model_params['d_y'])
    assert y_pred.dtype == valid_x.dtype
    assert y_pred.device == valid_x.device
    # Check that some predictions were made (not all zeros after initial steps)
    required_steps = max(model_params['d_i'], model_params['d_o'])
    if num_steps > required_steps:
         assert not torch.allclose(y_pred[:, required_steps:, :], torch.zeros_like(y_pred[:, required_steps:, :]))


def test_forward_open_mode(narx_model, valid_x, valid_y, model_params, batch_size, num_steps):
    """Tests forward pass in 'open' mode using true y."""
    y_pred = narx_model(valid_x, mode="open", y=valid_y)
    assert isinstance(y_pred, torch.Tensor)
    assert y_pred.shape == (batch_size, num_steps, model_params['d_y'])
    assert y_pred.dtype == valid_x.dtype
    assert y_pred.device == valid_x.device
     # Check that some predictions were made (not all zeros after initial steps)
    required_steps = max(model_params['d_i'], model_params['d_o'])
    if num_steps > required_steps:
        assert not torch.allclose(y_pred[:, required_steps:, :], torch.zeros_like(y_pred[:, required_steps:, :]))


def test_forward_close_mode_with_bootstrap(narx_model, valid_x, valid_y, model_params, batch_size, num_steps):
    """Tests forward pass in 'close' mode with bootstrapping."""
    required_steps = max(model_params['d_i'], model_params['d_o'])
    bootstrap_steps = required_steps + 2 # Example bootstrap value
    if bootstrap_steps >= num_steps:
         pytest.skip("Sequence length too short for meaningful bootstrap test")

    y_pred = narx_model(valid_x, mode="close", y=valid_y, bootstrap=bootstrap_steps)

    assert isinstance(y_pred, torch.Tensor)
    assert y_pred.shape == (batch_size, num_steps, model_params['d_y'])
    assert y_pred.dtype == valid_x.dtype
    assert y_pred.device == valid_x.device

    # --- Key check: Initial steps should match the provided y ---
    assert torch.allclose(y_pred[:, :bootstrap_steps, :], valid_y[:, :bootstrap_steps, :])

    # Check that predictions were made *after* the bootstrap period
    if num_steps > bootstrap_steps:
         assert not torch.allclose(y_pred[:, bootstrap_steps:, :], torch.zeros_like(y_pred[:, bootstrap_steps:, :]))
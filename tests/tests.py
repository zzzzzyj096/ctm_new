import torch
import pytest
import itertools
from models.constants import VALID_NEURON_SELECT_TYPES, VALID_BACKBONE_TYPES, VALID_POSITIONAL_EMBEDDING_TYPES
import numpy as np

def rep_size(neuron_select_type: str, n_synch: int) -> int:
    return n_synch if neuron_select_type == "random-pairing" else n_synch * (n_synch + 1) // 2

def rep_size(neuron_select_type: str, n_synch: int) -> int:
    return n_synch if neuron_select_type == "random-pairing" else n_synch * (n_synch + 1) // 2

def grab_synch_tensors(model, s_type: str):
    if s_type == "out":
        return (
            model.out_neuron_indices_left,
            model.out_neuron_indices_right,
            model.decay_params_out,
        )
    if s_type == "action":
        return (
            model.action_neuron_indices_left,
            model.action_neuron_indices_right,
            model.decay_params_action,
        )
    raise ValueError(s_type)

# --- Golden Tests --- 

def test_golden_parity(golden_test_model_parity, golden_test_input_parity, golden_test_expected_predictions_parity, golden_test_expected_certainties_parity, golden_test_expected_synchronization_out_tracking_parity, golden_test_expected_synchronization_action_tracking_parity, golden_test_expected_pre_activations_tracking_parity, golden_test_expected_post_activations_tracking_parity, golden_test_expected_attentions_tracking_parity):
    """Golden test the parity CTM model."""

    atol = 1e-5
    atol_attn = 1e-3
    golden_test_model_parity.eval()
    predictions, certainties, (synch_out_tracking, synch_action_tracking), pre_activations_tracking, post_activations_tracking, attention_tracking = golden_test_model_parity(golden_test_input_parity, track=True)

    assert torch.isclose(predictions, golden_test_expected_predictions_parity, atol=atol).all(), f"Predictions do not match expected values."
    assert torch.isclose(certainties, golden_test_expected_certainties_parity, atol=atol).all(), f"Certainties do not match expected values."
    assert np.isclose(synch_out_tracking, golden_test_expected_synchronization_out_tracking_parity, atol=atol).all(), f"Synch Out do not match expected values."
    assert np.isclose(synch_action_tracking, golden_test_expected_synchronization_action_tracking_parity, atol=atol).all(), f"Synch Action do not match expected values."
    assert np.isclose(pre_activations_tracking, golden_test_expected_pre_activations_tracking_parity, atol=atol).all(), f"Pre-activations do not match expected values."
    assert np.isclose(post_activations_tracking, golden_test_expected_post_activations_tracking_parity, atol=atol).all(), f"Post-activations do not match expected values."
    assert np.isclose(attention_tracking, golden_test_expected_attentions_tracking_parity, atol=atol_attn).all(), f"Attention do not match expected values."

    pass

def test_golden_qamnist(golden_test_model_qamnist, golden_test_input_qamnist, golden_test_expected_predictions_qamnist, golden_test_expected_certainties_qamnist, golden_test_expected_synchronization_out_tracking_qamnist, golden_test_expected_pre_activations_tracking_qamnist, golden_test_expected_post_activations_tracking_qamnist, golden_test_expected_attentions_tracking_qamnist, golden_test_expected_embeddings_tracking_qamnist):
    """Golden test the QAMNIST CTM model."""
    
    atol = 1e-4
    atol_attn = 5e-3
    golden_test_model_qamnist.eval()
    x, z = golden_test_input_qamnist

    predictions, certainties, synch_out_tracking, pre_activations_tracking, post_activations_tracking, attention_tracking, embedding_tracking = golden_test_model_qamnist(x, z=z, track=True)

    assert torch.isclose(predictions, golden_test_expected_predictions_qamnist, atol=atol).all(), f"Predictions do not match expected values."
    assert torch.isclose(certainties, golden_test_expected_certainties_qamnist, atol=atol).all(), f"Certainties do not match expected values."
    assert torch.isclose(synch_out_tracking, golden_test_expected_synchronization_out_tracking_qamnist[-1], atol=atol).all(), f"Synch Out do not match expected values."
    assert np.isclose(pre_activations_tracking, golden_test_expected_pre_activations_tracking_qamnist, atol=atol).all(), f"Pre-activations do not match expected values."
    assert np.isclose(post_activations_tracking, golden_test_expected_post_activations_tracking_qamnist, atol=atol).all(), f"Post-activations do not match expected values."
    assert np.isclose(attention_tracking, golden_test_expected_attentions_tracking_qamnist, atol=atol_attn).all(), f"Attention do not match expected values."
    assert np.isclose(embedding_tracking, golden_test_expected_embeddings_tracking_qamnist, atol=atol).all(), f"Embeddings do not match expected values."

    pass

def test_golden_rl(golden_test_model_rl, golden_test_inputs_rl, golden_test_expected_initial_state_trace_rl, golden_test_expected_initial_activated_state_trace_rl, golden_test_expected_action_rl, golden_test_expected_action_log_prob_rl, golden_test_expected_action_entropy_rl, golden_test_expected_value_rl, golden_test_expected_state_trace_rl, golden_test_expected_activated_state_trace_rl, golden_test_expected_action_logits_rl, golden_test_expected_action_probs_rl, golden_test_expected_pre_activations_tracking_rl, golden_test_expected_post_activations_tracking_rl, golden_test_expected_synch_out_tracking_rl):

    atol = 1e-5
    golden_test_model_rl.eval()

    initial_state_trace, initial_activated_state_trace = golden_test_model_rl.get_initial_state(num_envs=1)

    dones = torch.zeros(1).to(initial_state_trace.device)

    assert torch.isclose(initial_state_trace, golden_test_expected_initial_state_trace_rl, atol=atol).all(), f"Initial hidden states of the CTM does not match expected values."
    assert torch.isclose(initial_activated_state_trace, golden_test_expected_initial_activated_state_trace_rl, atol=atol).all(), f"Initial hidden states of the CTM does not match expected values."

    _, action_log_probs, entropy, value, (state_trace, activated_state_trace), tracking_data, action_logits, action_probs = golden_test_model_rl.get_action_and_value(golden_test_inputs_rl, (initial_state_trace, initial_activated_state_trace), dones, track=True)

    pre_activations = tracking_data["pre_activations"]
    post_activations = tracking_data["post_activations"]
    synchronization = tracking_data["synchronisation"]

    assert torch.isclose(action_log_probs, golden_test_expected_action_log_prob_rl, atol=atol).all(), f"Action log probs do not match expected values."
    assert torch.isclose(entropy, golden_test_expected_action_entropy_rl, atol=atol).all(), f"Entropy does not match expected values."
    assert torch.isclose(value, golden_test_expected_value_rl, atol=atol).all(), f"Value does not match expected values."
    assert torch.isclose(state_trace, golden_test_expected_state_trace_rl, atol=atol).all(), f"State trace does not match expected values."
    assert torch.isclose(activated_state_trace, golden_test_expected_activated_state_trace_rl, atol=atol).all(), f"Activated state trace does not match expected values."
    assert np.isclose(pre_activations, golden_test_expected_pre_activations_tracking_rl, atol=atol).all(), f"Pre-activations do not match expected values."
    assert np.isclose(post_activations, golden_test_expected_post_activations_tracking_rl, atol=atol).all(), f"Post-activations do not match expected values."
    assert np.isclose(synchronization, golden_test_expected_synch_out_tracking_rl, atol=atol).all(), f"Synchronisation do not match expected values."
    assert torch.isclose(action_logits, golden_test_expected_action_logits_rl, atol=atol).all(), f"Action logits do not match expected values."
    assert torch.isclose(action_probs, golden_test_expected_action_probs_rl, atol=atol).all(), f"Action probs do not match expected values."

    pass

# --- General CTM Tests ---

@pytest.mark.parametrize("synch_type", ["out", "action"])
@pytest.mark.parametrize("neuron_select_type", ["first-last", "random", "random-pairing"])
def test_set_synchronisation_parameters(ctm_factory, base_params, device, synch_type, neuron_select_type):
    np.random.seed(0)
    n_synch = 8
    num_random_pairing_self = 2

    model = ctm_factory(
        base_params,
        d_model=64,
        n_synch_out=n_synch,
        n_synch_action=n_synch,
        neuron_select_type=neuron_select_type,
        n_random_pairing_self=num_random_pairing_self,
    ).to(device)

    left, right, decay = grab_synch_tensors(model, synch_type)

    # Check shapes
    assert left.dtype == right.dtype == torch.long
    assert left.shape == right.shape == (n_synch,)
    assert decay.shape == (rep_size(neuron_select_type, n_synch),)

    # Check equal number of neurons on left and right
    assert left.size(0) == right.size(0) == n_synch
    # Check that the left and right indices are within the model's d_model
    assert torch.all(left < model.d_model) and torch.all(right < model.d_model)

    # Test neuron pairing selection
    if neuron_select_type == "first-last":
        if synch_type == "out":
            exp = torch.arange(0, n_synch, device=device)
        else:
            exp = torch.arange(model.d_model - n_synch, model.d_model, device=device)
        assert torch.equal(left, exp) and torch.equal(right, exp)
    elif neuron_select_type == "random":
        pass
    elif neuron_select_type == "random-pairing":
        assert torch.equal(right[:num_random_pairing_self], left[:num_random_pairing_self])

# ------ Neuron Select Type Test ---

@pytest.mark.parametrize("neuron_select_type", VALID_NEURON_SELECT_TYPES)
def test_valid_neuron_select_type(ctm_factory, base_params, neuron_select_type):
    model = ctm_factory(base_params, neuron_select_type=neuron_select_type)
    assert model is not None

def test_none_neuron_select_type(ctm_factory, base_params):
    with pytest.raises(Exception):
        ctm_factory(base_params, neuron_select_type="none")

def test_invalid_neuron_select_type(ctm_factory, base_params):
    with pytest.raises(Exception):
        ctm_factory(base_params, neuron_select_type="invalid-option")

# ------ Backbone and Positional Embedding Type Test ---

@pytest.mark.parametrize("backbone_type, positional_embedding_type", list(itertools.product(VALID_BACKBONE_TYPES, VALID_POSITIONAL_EMBEDDING_TYPES)))
def test_valid_backbone_and_valid_positional_embedding(ctm_factory, base_params, backbone_type, positional_embedding_type):
    model = ctm_factory(
        base_params,
        backbone_type=backbone_type,
        positional_embedding_type=positional_embedding_type,
    )
    assert model is not None

def test_none_backbone_with_none_positional_embeddings(ctm_factory, base_params):
    model = ctm_factory(
        base_params,
        backbone_type="none",
        positional_embedding_type="none",
    )
    assert model is not None

@pytest.mark.parametrize("positional_embedding_type", VALID_POSITIONAL_EMBEDDING_TYPES)
def test_none_backbone_with_valid_positional_embeddings(ctm_factory, base_params, positional_embedding_type):
    with pytest.raises(Exception):
        ctm_factory(
            base_params,
            backbone_type="none",
            positional_embedding_type=positional_embedding_type,
        )

@pytest.mark.parametrize("backbone_type", VALID_BACKBONE_TYPES)
def test_valid_backbone_with_none_positional_embeddings(ctm_factory, base_params, backbone_type):
    model = ctm_factory(
        base_params,
        backbone_type=backbone_type,
        positional_embedding_type="none",
    )
    assert model is not None

# --- Parity Tests ---

def test_parity_prediction_shape(parity_ctm_model, parity_params, parity_input):
    predictions, _, _ = parity_ctm_model(parity_input)

    batch_size, parity_length = parity_input.shape
    expected_shape = (batch_size, parity_length * 2, parity_params["iterations"])
    assert predictions.shape == expected_shape

def test_parity_certainty_shape(parity_ctm_model, parity_params, parity_input):
    _, certainties, _ = parity_ctm_model(parity_input)

    batch_size = parity_input.shape[0]
    expected_shape = (batch_size, 2, parity_params["iterations"])
    assert certainties.shape == expected_shape

def test_parity_nans_in_predictions(parity_ctm_model, parity_input):
    predictions, _, _ = parity_ctm_model(parity_input)
    assert not torch.isnan(predictions).any()

# --- QAMNIST Tests ---

def test_qamnist_prediction_shape(qamnist_model_factory, qamnist_params, qamnist_input, device):
    model = qamnist_model_factory("first-last").to(device)
    inputs, z = qamnist_input

    predictions, _, _ = model(inputs, z)
    B = inputs.shape[0]
    out_dims = qamnist_params["out_dims"]
    T = inputs.shape[1] + z.shape[1] + qamnist_params["iterations_for_answering"]
    expected_shape = (B, out_dims, T)
    assert predictions.shape == expected_shape, f"Expected {expected_shape}, got {predictions.shape}"

def test_qamnist_certainty_shape(qamnist_model_factory, qamnist_params, qamnist_input, device):
    model = qamnist_model_factory("first-last").to(device)
    inputs, z = qamnist_input

    _, certainties, _ = model(inputs, z)
    B = inputs.shape[0]
    T = inputs.shape[1] + z.shape[1] + qamnist_params["iterations_for_answering"]
    expected_shape = (B, 2, T)
    assert certainties.shape == expected_shape, f"Expected {expected_shape}, got {certainties.shape}"

def test_qamnist_nans_in_predictions(qamnist_model_factory, qamnist_input, device):
    model = qamnist_model_factory("first-last").to(device)
    inputs, z = qamnist_input

    predictions, _, _ = model(inputs, z)
    assert not torch.isnan(predictions).any(), "Predictions contain NaNs"

@pytest.mark.parametrize("neuron_select_type", ["first-last", "random", "random-pairing"])
def test_qamnist_synchronisation_shape(qamnist_model_factory, qamnist_params, qamnist_input, neuron_select_type, device):
    model = qamnist_model_factory(neuron_select_type).to(device)
    inputs, z = qamnist_input

    _, _, synchronisation = model(inputs, z)

    batch_size = inputs.shape[0]
    n_synch_out = qamnist_params["n_synch_out"]

    if neuron_select_type in ("first-last", "random"):
        expected_size = (n_synch_out * (n_synch_out + 1)) // 2
    elif neuron_select_type == "random-pairing":
        expected_size = n_synch_out

    assert synchronisation.shape == (batch_size, expected_size), \
        f"Expected {(batch_size, expected_size)}, got {synchronisation.shape}"

import pytest
import torch
from models.ctm import ContinuousThoughtMachine
from models.ctm_qamnist import ContinuousThoughtMachineQAMNIST
from utils.samplers import QAMNISTSampler
from tasks.qamnist.utils import get_dataset
from tests.test_data import *
from utils.housekeeping import set_seed
from tasks.rl.train import Agent
from types import SimpleNamespace

# --- Housekeeping ---

@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def seed():
    return 42

@pytest.fixture(autouse=True)
def auto_set_seed(seed):
    set_seed(seed)

# --- Golden Test Fixtures ---

# ------ Parity ------

@pytest.fixture
def golden_test_params_parity(parity_params):
    parity_length = 4
    parity_small_params = parity_params.copy()
    parity_small_params["out_dims"] = 2 * parity_length
    parity_small_params["prediction_reshaper"] = [parity_length, 2]
    return parity_small_params

@pytest.fixture
def golden_test_model_parity(golden_test_params_parity, device):
    return ContinuousThoughtMachine(**golden_test_params_parity).to(device)

@pytest.fixture
def golden_test_input_parity(device):
    return GOLDEN_TEST_INPUT_PARITY.to(device)

@pytest.fixture
def golden_test_expected_predictions_parity(device):
    return GOLDEN_TEST_EXPECTED_PREDICTIONS_PARITY.to(device)

@pytest.fixture
def golden_test_expected_certainties_parity(device):
    return GOLDEN_TEST_EXPECTED_CERTAINTIES_PARITY.to(device)

@pytest.fixture
def golden_test_expected_synchronization_out_tracking_parity():
    return GOLDEN_TEST_EXPECTED_SYNCH_OUT_TRACKING_PARITY

@pytest.fixture
def golden_test_expected_synchronization_action_tracking_parity():
    return GOLDEN_TEST_EXPECTED_SYNCH_ACTION_TRACKING_PARITY

@pytest.fixture
def golden_test_expected_pre_activations_tracking_parity():
    return GOLDEN_TEST_EXPECTED_PRE_ACTIVATIONS_TRACKING_PARITY

@pytest.fixture
def golden_test_expected_post_activations_tracking_parity():
    return GOLDEN_TEST_EXPECTED_POST_ACTIVATIONS_TRACKING_PARITY

@pytest.fixture
def golden_test_expected_attentions_tracking_parity():
    return GOLDEN_TEST_EXPECTED_ATTENTIONS_TRACKING_PARITY

# ------ QAMNIST ------

@pytest.fixture
def golden_test_params_qamnist(base_params):
    params = base_params.copy()
    params.pop("backbone_type")
    params.pop("positional_embedding_type")
    params["iterations_per_digit"] = 1
    params["iterations_per_question_part"] = 1
    params["iterations_for_answering"] = 1
    return params

@pytest.fixture
def golden_test_model_qamnist(golden_test_params_qamnist, device):
    return ContinuousThoughtMachineQAMNIST(**golden_test_params_qamnist).to(device)

@pytest.fixture
def golden_test_input_qamnist(device):
    q_num_images = 2
    q_num_images_delta = 0
    q_num_repeats_per_input = 1
    q_num_operations = 2
    q_num_operations_delta = 0
    batch_size = 1

    train_data, _, _, _, _ = get_dataset(
        q_num_images=q_num_images,
        q_num_images_delta=q_num_images_delta,
        q_num_repeats_per_input=q_num_repeats_per_input,
        q_num_operations=q_num_operations,
        q_num_operations_delta=q_num_operations_delta,
    )

    sampler = QAMNISTSampler(train_data, batch_size=batch_size)
    loader = torch.utils.data.DataLoader(train_data, batch_sampler=sampler, num_workers=0)

    inputs, z, _, _ = next(iter(loader))
    inputs = inputs.to(device)
    z = torch.stack(z, 1).to(device)

    return inputs, z

@pytest.fixture
def golden_test_expected_predictions_qamnist(device):
    return GOLDEN_TEST_EXPECTED_PREDICTIONS_QAMNIST.to(device)

@pytest.fixture
def golden_test_expected_certainties_qamnist(device):
    return GOLDEN_TEST_EXPECTED_CERTAINTIES_QAMNIST.to(device)

@pytest.fixture
def golden_test_expected_synchronization_out_tracking_qamnist(device):
    return GOLDEN_TEST_EXPECTED_SYNCH_OUT_TRACKING_QAMNIST.to(device)

@pytest.fixture
def golden_test_expected_synchronization_action_tracking_qamnist():
    return GOLDEN_TEST_EXPECTED_SYNCH_ACTION_TRACKING_QAMNIST

@pytest.fixture
def golden_test_expected_pre_activations_tracking_qamnist():
    return GOLDEN_TEST_EXPECTED_PRE_ACTIVATIONS_TRACKING_QAMNIST

@pytest.fixture
def golden_test_expected_post_activations_tracking_qamnist():
    return GOLDEN_TEST_EXPECTED_POST_ACTIVATIONS_TRACKING_QAMNIST

@pytest.fixture
def golden_test_expected_attentions_tracking_qamnist():
    return GOLDEN_TEST_EXPECTED_ATTENTIONS_TRACKING_QAMNIST

@pytest.fixture
def golden_test_expected_embeddings_tracking_qamnist():
    return GOLDEN_TEST_EXPECTED_EMBEDDINGS_TRACKING_QAMNIST

# ------ RL (CartPole) ------

@pytest.fixture
def golden_test_params_rl(base_params):
    params = base_params.copy()
    params.pop("heads")
    params.pop("n_synch_action")
    params.pop("out_dims")
    params.pop("n_random_pairing_self")
    params.pop("positional_embedding_type")
    return params

@pytest.fixture
def golden_test_model_rl(golden_test_params_rl, device):
    args = SimpleNamespace(
        env_id="CartPole-v1",
        model_type="ctm",
        continuous_state_trace=True,
        iterations=golden_test_params_rl['iterations'],
        d_model=golden_test_params_rl['d_model'],
        d_input=golden_test_params_rl['d_input'],
        n_synch_out=golden_test_params_rl['n_synch_out'],
        synapse_depth=golden_test_params_rl['synapse_depth'],
        memory_length=golden_test_params_rl['memory_length'],
        deep_memory=golden_test_params_rl['deep_nlms'],
        memory_hidden_dims=golden_test_params_rl['memory_hidden_dims'],
        do_normalisation=golden_test_params_rl['do_layernorm_nlm'],
        dropout=golden_test_params_rl.get('dropout', 0),
        neuron_select_type=golden_test_params_rl.get('neuron_select_type', 'first-last'),
    )
    size_action_space = 2  
    model = Agent(size_action_space, args, device).to(device)
    return model

@pytest.fixture()
def environment_to_test():
    return "cartpole"

@pytest.fixture
def golden_test_inputs_rl(environment_to_test):
    if environment_to_test != "cartpole":
        raise NotImplementedError("RL test only tests cartpole.")
    observations = torch.tensor([[ 0.01,  0.02,  0.03,  0.04],], dtype=torch.float32)
    return observations

@pytest.fixture
def golden_test_expected_initial_state_trace_rl(device):
    return GOLDEN_TEST_EXPECTED_INITIAL_STATE_TRACE_RL.to(device)

@pytest.fixture
def golden_test_expected_initial_activated_state_trace_rl(device):
    return GOLDEN_TEST_EXPECTED_INITIAL_ACTIVATED_STATE_TRACE_RL.to(device)

@pytest.fixture
def golden_test_expected_action_rl(device):
    return GOLDEN_TEST_EXPECTED_ACTION_RL.to(device)

@pytest.fixture
def golden_test_expected_action_log_prob_rl(device):
    return GOLDEN_TEST_EXPECTED_ACTION_LOG_PROB_RL.to(device)

@pytest.fixture
def golden_test_expected_action_entropy_rl(device):
    return GOLDEN_TEST_EXPECTED_ENTROPY_RL.to(device)

@pytest.fixture
def golden_test_expected_value_rl(device):
    return GOLDEN_TEST_EXPECTED_VALUE_RL.to(device)

@pytest.fixture
def golden_test_expected_state_trace_rl(device):
    return GOLDEN_TEST_EXPECTED_STATE_TRACE_RL.to(device)

@pytest.fixture
def golden_test_expected_activated_state_trace_rl(device):
    return GOLDEN_TEST_EXPECTED_ACTIVATED_STATE_TRACE_RL.to(device)

@pytest.fixture
def golden_test_expected_action_logits_rl(device):
    return GOLDEN_TEST_EXPECTED_ACTION_LOGITS_RL.to(device)

@pytest.fixture
def golden_test_expected_action_probs_rl(device):
    return GOLDEN_TEST_EXPECTED_ACTION_PROBS_RL.to(device)

@pytest.fixture
def golden_test_expected_pre_activations_tracking_rl():
    return GOLDEN_TEST_EXPECTED_PRE_ACTIVATIONS_TRACKING_RL

@pytest.fixture
def golden_test_expected_post_activations_tracking_rl():
    return GOLDEN_TEST_EXPECTED_POST_ACTIVATIONS_TRACKING_RL

@pytest.fixture
def golden_test_expected_synch_out_tracking_rl():
    return GOLDEN_TEST_SYNCH_OUT_TRACKING_RL

# --- Parity Fixtures ---

@pytest.fixture
def base_params():
    return dict(
        iterations=10,
        d_model=32,
        d_input=4,
        heads=2,
        n_synch_out=3,
        n_synch_action=3,
        synapse_depth=1,
        memory_length=5,
        deep_nlms=True,
        memory_hidden_dims=2,
        do_layernorm_nlm=False,
        backbone_type="none",
        positional_embedding_type="none",
        out_dims=10,
        prediction_reshaper=[-1],
        dropout=0.0,
        neuron_select_type="first-last",
        n_random_pairing_self=0,
    )

@pytest.fixture
def parity_input(device):
    batch_size = 4
    parity_length = 64
    return torch.randint(0, 2, (batch_size, parity_length), dtype=torch.float32, device=device) * 2 - 1

@pytest.fixture
def ctm_factory():
    def _create_model(base_config, **overrides):
        config = base_config.copy()
        config.update(overrides)
        return ContinuousThoughtMachine(**config)
    return _create_model

@pytest.fixture
def parity_params(base_params):
    parity_length = 64
    parity_params = base_params.copy()
    parity_params["backbone_type"] = "parity_backbone"
    parity_params["positional_embedding_type"] = "custom-rotational-1d"
    parity_params["out_dims"] = 2 * parity_length
    parity_params["prediction_reshaper"] = [parity_length, 2]
    return parity_params

@pytest.fixture
def parity_ctm_model(parity_params, device):
    return ContinuousThoughtMachine(**parity_params).to(device)

# --- QAMNIST Fixtures ---

@pytest.fixture
def qamnist_params():
    return dict(
        iterations=1,
        d_model=1024,
        d_input=64,
        heads=4,
        n_synch_out=32,
        n_synch_action=32,
        synapse_depth=1,
        memory_length=10,
        deep_nlms=True,
        memory_hidden_dims=16,
        do_layernorm_nlm=True,
        out_dims=10,
        prediction_reshaper=[-1],
        dropout=0.0,
        neuron_select_type="first-last",
        iterations_per_digit=10,
        iterations_per_question_part=10,
        iterations_for_answering=10,
        n_random_pairing_self=0,
    )

@pytest.fixture
def qamnist_input(device):
    q_num_images = 3
    q_num_images_delta = 2
    q_num_repeats_per_input = 10
    q_num_operations = 3
    q_num_operations_delta = 2
    batch_size = 4

    train_data, _, _, _, _ = get_dataset(
        q_num_images=q_num_images,
        q_num_images_delta=q_num_images_delta,
        q_num_repeats_per_input=q_num_repeats_per_input,
        q_num_operations=q_num_operations,
        q_num_operations_delta=q_num_operations_delta,
    )

    sampler = QAMNISTSampler(train_data, batch_size=batch_size)
    loader = torch.utils.data.DataLoader(train_data, batch_sampler=sampler, num_workers=0)

    inputs, z, _, _ = next(iter(loader))
    inputs = inputs.to(device)
    z = torch.stack(z, 1).to(device)

    return inputs, z

@pytest.fixture
def qamnist_model_factory(qamnist_params):
    def _create_model(neuron_select_type):
        return ContinuousThoughtMachineQAMNIST(
            **{**qamnist_params, "neuron_select_type": neuron_select_type}
        )
    return _create_model


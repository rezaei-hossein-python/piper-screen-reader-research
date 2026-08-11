import sys
from pathlib import Path
from unittest.mock import MagicMock
import torch

# Create a mock for monotonic_align so we can run forward-pass shape and gradient tests
# without having to compile Cython on the local CPU machine!
batch_size = 2
dummy_input_length = 15
dummy_spec_length = 30

mock_mono = MagicMock()
dummy_attn = torch.zeros(batch_size, dummy_spec_length, dummy_input_length)
for b in range(batch_size):
    for i in range(min(dummy_spec_length, dummy_input_length)):
        dummy_attn[b, i, i] = 1.0

mock_mono.maximum_path.return_value = dummy_attn
sys.modules['piper.train.vits.monotonic_align'] = mock_mono

UPSTREAM_PATH = Path("C:/projects/piper-screen-reader-research/upstream/piper/src")
sys.path.insert(0, str(UPSTREAM_PATH))

from piper.train.vits.models import SynthesizerTrn

def test_shape_and_gradients():
    print("Initializing test_shape_and_gradients...")
    
    # Initialize the model
    model = SynthesizerTrn(
        n_vocab=100,
        spec_channels=513,
        segment_size=8192,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        resblock="1",
        resblock_kernel_sizes=(3, 5, 7),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        upsample_rates=(8, 8, 2, 2),
        upsample_initial_channel=512,
        upsample_kernel_sizes=(16, 16, 4, 4),
        n_speakers=1,
        gin_channels=256,
        use_sdp=True,
    )
    
    print("Model initialized successfully!")
    
    # Let's initialize the emb_mode weight with non-zero values
    torch.nn.init.normal_(model.emb_mode.weight, std=0.1)
    
    # To prove our zero-projection gradient blocking hypothesis, we will temporarily
    # initialize the projection weights of the flows to non-zero values!
    # (In real training, as soon as other parts of the network update, or because we use an optimizer,
    # the weights become non-zero, or we can train with standard PyTorch initialization.
    # In VITS, they zero-init projection weights so that at step 0, the flow is an identity transform).
    for flow in model.dp.flows:
        if hasattr(flow, "proj"):
            torch.nn.init.normal_(flow.proj.weight, std=0.1)
            torch.nn.init.normal_(flow.proj.bias, std=0.1)
            
    for flow in model.dp.post_flows:
        if hasattr(flow, "proj"):
            torch.nn.init.normal_(flow.proj.weight, std=0.1)
            torch.nn.init.normal_(flow.proj.bias, std=0.1)
            
    x = torch.randint(low=0, high=100, size=(batch_size, dummy_input_length), dtype=torch.long)
    x_lengths = torch.LongTensor([dummy_input_length, dummy_input_length - 3])
    spec = torch.randn(batch_size, 513, dummy_spec_length)
    spec_lengths = torch.LongTensor([dummy_spec_length, dummy_spec_length - 5])
    speech_mode = torch.LongTensor([0, 1])
    
    outputs = model.forward(
        x,
        x_lengths,
        spec,
        spec_lengths,
        sid=None,
        speech_mode=speech_mode,
    )
    
    (o, l_length, attn, ids_slice, x_mask, y_mask, stats) = outputs
    loss_dur = torch.sum(l_length)
    
    model.zero_grad()
    loss_dur.backward()
    
    # Check gradients of self.dp.cond
    cond_grad = model.dp.cond.weight.grad
    cond_bias_grad = model.dp.cond.bias.grad
    print(f"dp.cond weight grad norm: {cond_grad.norm().item() if cond_grad is not None else 'None'}")
    
    # Check that gradients reach the emb_mode weight parameter
    emb_mode_grad = model.emb_mode.weight.grad
    print(f"emb_mode weight gradient norm: {emb_mode_grad.norm().item() if emb_mode_grad is not None else 'None'}")
    
    assert emb_mode_grad is not None, "Gradients did not flow to emb_mode!"
    assert torch.any(emb_mode_grad != 0), "Gradients are zero on emb_mode!"
    
    print("All shape and gradient tests passed successfully with non-zero flow projections!")

if __name__ == "__main__":
    try:
        test_shape_and_gradients()
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

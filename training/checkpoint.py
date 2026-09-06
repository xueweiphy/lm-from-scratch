"""Checkpointing (CS336 A1, checkpointing).

A checkpoint is one dict — model.state_dict(), optimizer.state_dict(), and the
training-loop iteration — written with torch.save.  The iteration is the loop's
own counter (the schedule's t), not something the model or optimizer tracks;
load_checkpoint hands it back so training resumes at the right step.
`out` / `src` may be a path or an open binary file.
"""

import torch


def save_checkpoint ( model , optimizer, iteration, out ) :
    state_dict = {}
    st1  = model.state_dict()
    st2 = optimizer.state_dict ()
    state_dict["model"] = st1
    state_dict["optimizer"] = st2
    state_dict["iteration"] = iteration
    torch.save ( state_dict, out )


def load_checkpoint ( src, model, optimizer ) :
    state_dict = torch.load ( src )
    model.load_state_dict ( state_dict["model"] )
    optimizer.load_state_dict ( state_dict["optimizer"] )
    return state_dict["iteration"]

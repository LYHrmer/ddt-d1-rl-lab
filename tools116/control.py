"""Add the existing D1 encoder trim to the speed selected by the lidar preview."""
import numpy as np

from tools91.launch91 import require
from tools110.course import command
from tools113.control import BrakeTrim


class PrevTrimActor:
    def __init__(self, session, read_wheels, enabled=True):
        self.session = session
        self.read_wheels = read_wheels
        self.enabled = enabled
        self.control = BrakeTrim()
        self.last_actual = 0.
        self._preview = None
        self.commands = []
        self.increments = []
        self.encoders = []
        self.previous_commands = []
        self.previews = []
        self.raw_feedback = []
        self.history_feedback = []
        self.history_commands = []

    def set_preview(self, vx):
        """Receive this tick's unrounded return from Preview.targets before actor_frame."""
        require(self._preview is None and np.isfinite(vx), "missing or duplicated lidar preview clock")
        self._preview = float(vx)

    def run(self, outputs, inputs):
        current, history = inputs["nn_input0"], inputs["nn_input1"]
        preview = self._preview
        require(preview is not None and current.shape == (1,57) and history.shape == (1,10,57)
                and current.dtype == history.dtype == np.float32 and current[0,6] == np.float32(2.*preview),
                "actor frame does not correspond to the pending lidar preview")
        self._preview = None
        request = command(len(self.commands))
        increment = self.control.update(request,self.last_actual,self.read_wheels())
        applied_increment = increment if self.enabled else 0.
        actual = preview + applied_increment
        if applied_increment:
            current[0,6] = np.float32(2.*actual)
        # Match R113: retain the full-precision command whose float32 encoding went to the actor.
        self.previous_commands.append(self.last_actual)
        self.last_actual = actual
        self.commands.append(actual)
        self.previews.append(preview)
        self.increments.append(applied_increment)
        self.encoders.append(self.control.encoder_forward_m_s)
        self.raw_feedback.append(current[0,41:57].copy())
        self.history_feedback.append(history[0,:,41:57].copy())
        self.history_commands.append(history[0,:,6:9].copy())
        return self.session.run(outputs,inputs)

    def trace(self, rows):
        """Preserve all attempted inputs even if the last native physics interval failed."""
        trace = {key:np.asarray(values) for key,values in rows.items()}
        trace.update(preview_actor_vx=np.asarray(self.previews), actor_vx=np.asarray(self.commands),
                     velocity_trim=np.asarray(self.increments), encoder_forward_m_s=np.asarray(self.encoders),
                     trim_previous_actor_vx=np.asarray(self.previous_commands),
                     actor_previous_raw=np.asarray(self.raw_feedback),
                     actor_history_previous_raw=np.asarray(self.history_feedback),
                     actor_history_frame_command=np.asarray(self.history_commands))
        return trace

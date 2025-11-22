from queue import Queue
from queue import Empty
import pickle
import onnxruntime as ort
import numpy as np

import global_vars
from .base import ModelBase
import inference_vars


class Step(ModelBase):
    def __init__(self, model_path, state_path, dt=None):
        super().__init__()
        self.model_path = model_path
        self.state_path = state_path
        self.model = ort.InferenceSession(model_path)
        with open(state_path, "rb") as f:
            self.state = pickle.load(f)
        self.dt = np.array(dt).astype("float16")
        self.last_timestamp = None
        inference_vars.inference_completed = False

    def __call__(self, preprocess_queue: Queue, result_queue: Queue):
        while not inference_vars.inference_completed and not global_vars.user_interrupt:
            try:
                frame, timestamp = preprocess_queue.get(timeout=1)
                if self.last_timestamp is None:
                    self.last_timestamp = timestamp
                    dt = self.dt
                else:
                    dt = timestamp - self.last_timestamp
                    dt = np.array(dt).astype("float16")
                    self.last_timestamp = timestamp
                    
            except Empty:
                if inference_vars.preprocess_completed:
                    inference_vars.inference_completed = True
                break
                
            image = np.array([[frame]]).astype("float16") / 255.0
            input_dict = {"arg_0.1": image, "onnx::Mul_37": dt, **self.state}
            result = self.model.run(None, input_dict)
            self.state = dict(zip(list(input_dict)[2:], result[1:]))
            result_queue.put((result[0][0, 0], timestamp))
        with open(self.state_path, "wb") as f:
            pickle.dump(self.state, f)

        self.last_timestamp = None
            
        inference_vars.inference_completed = True

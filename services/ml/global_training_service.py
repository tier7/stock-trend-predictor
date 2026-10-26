import logging
from copy import deepcopy
from threading import Lock, Thread
from uuid import uuid4

from services.ml.global_forest_service import train_global_model
from services.ml.data_preparation_service import validate_training_cutoff


class GlobalTrainingJobs:

    def __init__(self):
        self._lock = Lock()
        self._state = None
        self._thread = None

    def snapshot(self):
        with self._lock:
            return deepcopy(self._state)

    def start(self, training_cutoff, model_dir=None):
        validate_training_cutoff(training_cutoff)
        with self._lock:
            if self._state and self._state["status"] == "running":
                raise RuntimeError("Trening globalnego modelu już trwa.")
            self._state = {
                "job_id": uuid4().hex,
                "status": "running",
                "training_cutoff": training_cutoff,
                "message": "Rozpoczynanie przygotowania danych treningowych.",
            }
            initial_state = deepcopy(self._state)
        self._thread = Thread(
            target=self._run, args=(training_cutoff, model_dir),
            name="global-forest-training", daemon=True,
        )
        self._thread.start()
        return initial_state

    def _progress(self, message):
        with self._lock:
            self._state["message"] = message

    def _run(self, training_cutoff, model_dir):
        try:
            metadata = train_global_model(training_cutoff, model_dir, self._progress)
        except Exception as error:
            logging.getLogger(__name__).exception("Global forest training failed")
            message = str(error) if isinstance(error, ValueError) else (
                "Trening nie powiódł się. Szczegóły znajdują się w logu serwera."
            )
            with self._lock:
                self._state.update(status="failed", message=message)

        else:
            with self._lock:
                self._state.update(
                    status="completed", message="Model został wytrenowany i zapisany.",
                    model_metadata=metadata)

import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression # type: ignore

class Model(ABC):
    """
    Abstract base class for all models.
    """

    @abstractmethod
    def train(self, x_train, y_train):
        """
        Trains the model on the given data.

        Args:
            x_train: Training data
            y_train: Target data
        """
        pass

class LinearRegressionModel(Model):
    """
     LinearRegressionModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(x_train, y_train)
            logging.info("Model trainig is completed")
            return reg
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e
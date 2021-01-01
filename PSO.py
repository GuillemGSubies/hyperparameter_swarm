from copy import deepcopy

import numpy as np
from sklearn.metrics import f1_score


class Particle:
    def __init__(self, position=None, velocities=None, score=0, personal_best=None):
        self.position = position or {}
        self.velocities = velocities or {}
        self.score = score
        self.personal_best = personal_best or {}


class PSO:
    """
    Hyperparameter search algorithm based on PSO.

    To use the algorithm; instance the class and call the `__call__` method without
    more parameters.

    Parameters
    ----------
    model : Sklearn-compatible estimator
        Simple sklearn estimator (only ClassifierMixin by the moment)
        uninstanced.
    params : dict
        Dictionary containing the parameters to be optimized and the ranges they can 
        take (only int or float, not categorical). The parameters will be sampled uniformly.
    X_train : {array-like, sparse matrix} of shape (n_samples, n_features)
        This estimator does not follow sklearn's API yet, so there is no internal,
        cross-validation, the test set here is used for calculating the f1 macro score and
        select the best model. In the future a custom metric could be provided.
    y_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
    X_valid : {array-like, sparse matrix} of shape (n_samples, n_features)
    y_valid : array-like of shape (n_samples,) or (n_samples, n_outputs)
    n_iter : int, default=50
        Number of iterations for the algorithm to run.
    n_particles : int, default=10
        Number of particles used.
    learning_rate : float, default=0.07
        Learning rate for the model.
    omega : float, default=0.1
        Omega parameter for the algorithm, velocity deceleration.
    fip : float, default=0.2
        Fi parameter of confidence in self for acceleration.
    fig : float, default=0.3,
        Fi parameter of confidence in neigbors for acceleration.
    n_jobs: int or  -1 deffault=-1
        Used only in the estimator, not the actual algorithm.
    random_state : int default=42
        Random state both for the estimator and the algorithm.  
    """

    def __init__(
        self,
        model,
        params,
        X_train,
        y_train,
        X_valid,
        y_valid,
        n_iter=50,
        n_particles=10,
        learning_rate=0.07,
        omega=0.1,
        fip=0.2,
        fig=0.3,
        n_jobs=-1,  # only for the model
        random_state=42,  # only for the model
    ):
        self.model = model
        self.params = params
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.n_iter = n_iter
        self.n_particles = n_particles
        self.learning_rate = learning_rate
        self.omega = omega  # acceleration
        self.fip = fip  # confidence on particle acceleration
        self.fig = fig  # confidence on neigbors acceleration
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.rand = np.random.RandomState(self.random_state )
        self.particles = []
        self.best_model = None
        self.best_particle = Particle()

    def __call__(self):
        self.initialize_particles()
        for _ in range(self.n_iter):
            for i, particle in enumerate(self.particles):
                for param in particle.position:
                    rp, rg = self.rand.uniform(0, 1), self.rand.uniform(0, 1)
                    current_vel = particle.velocities[param]
                    current_pos = particle.position[param]
                    personal_best = particle.personal_best[param]
                    general_best = self.best_particle.position[param]

                    new_velocity = (
                        self.omega * current_vel
                        + self.fip * rp * (personal_best - current_pos)
                        + self.fig * rg * (general_best - current_pos)
                    )
                    # sometimes its negative
                    new_pos = current_pos + self.learning_rate * new_velocity
                    new_pos = int(new_pos) if isinstance(current_pos, int) else new_pos
                    if new_pos > max(self.params[param]):
                        new_pos = max(self.params[param])
                    elif new_pos < min(self.params[param]):
                        new_pos = min(self.params[param])
                    particle.position[param] = abs(new_pos)

                self.particles[i] = deepcopy(self.compare(particle))
        return (
            self.best_model,
            self.best_particle.personal_best,
            self.best_particle.score,
        )

    def initialize_particles(self):
        for _ in range(self.n_particles):
            new_dist, velocities = {}, {}
            for param in self.params:
                lower, upper = self.params[param]
                if isinstance(lower, int):
                    rand = self.rand.randint
                else:
                    rand = self.rand.uniform
                new_dist[param] = rand(lower, upper)
                velocities[param] = rand(-abs(upper - lower), abs(upper - lower))
            particle = self.compare(Particle(new_dist, velocities))
            self.particles.append(particle)

    def compare(self, particle):
        try:
            model = self.model(
                **particle.position, n_jobs=self.n_jobs, random_state=self.random_state
            ).fit(self.X_train, self.y_train)
        except Exception:
            model = self.model(**particle.position).fit(self.X_train, self.y_train)
        score = f1_score(self.y_valid, model.predict(self.X_valid), average="macro")
        old_score = particle.score
        particle.score = score
        if particle.score > old_score:
            particle.personal_best = particle.position
        if particle.score > self.best_particle.score:
            self.best_particle = deepcopy(particle)
            self.best_model = model
        return particle

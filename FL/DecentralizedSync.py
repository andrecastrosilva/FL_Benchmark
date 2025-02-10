from Utils.FLUtils import FLUtils
import requests # WebInterface

class DecentralizedSync(FLUtils):
    
    def __init__(self, *,
        local_epochs = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.local_epochs = int(local_epochs)
        self.base_path += f'/{self.local_epochs}'


    def setup(self):
        self.ml.compile_model()

        if self.comm.is_master():
            self.x_val, self.y_val = self.ml.load_data('val')
            self.node_weights = [0] * self.comm.n_workers
            for _ in range(self.comm.n_workers):
                worker_id, n_samples = self.comm.recv_worker()
                self.node_weights[worker_id] = n_samples
            self.node_weights = [n_samples / sum(self.node_weights) for n_samples in self.node_weights]
            self.comm.send_workers(self.ml.get_weights())

        else:
            x_train, _ = self.ml.load_worker_data(self.comm.worker_id, self.comm.n_workers)
            n_samples = len(x_train)
            self.comm.send_master(n_samples)
            weights = self.comm.recv_master()
            self.ml.set_weights(weights)


    def master_train(self):

        data: dict = {'benchmarkProgressBarMaxValue': self.epochs,
                      'benchmarkProgressBar': 0,
                      'benchmarkStatus': True} # WebInterface

        for epoch in range(1, self.epochs+1):
            avg_weights = []
            for _ in range(self.comm.n_workers):
                worker_id, weights = self.comm.recv_worker()

                if avg_weights == []:
                    avg_weights = [w * self.node_weights[worker_id] for w in weights]
                else:
                    avg_weights = [a + w * self.node_weights[worker_id] for a, w in zip(avg_weights, weights)]
            
            self.ml.set_weights(avg_weights)
            self.comm.send_workers(avg_weights)
            new_score = self.validate(epoch)
            stop = self.early_stop(new_score) or epoch == self.epochs
            self.comm.send_workers(stop)

            data['benchmarkProgressBar'] = epoch # WebInterface
            requests.post('http://localhost:5000/benchmarkProgressBar', json=data) # WebInterface
            
            if stop:
                data['benchmarkStatus'] = False # WebInterface
                requests.post('http://localhost:5000/benchmarkProgressBar', json=data) # WebInterface
                break


    def worker_train(self):
        stop = False
        while not stop:
            self.ml.train(self.local_epochs)
            weights = self.ml.get_weights()
            self.comm.send_master(weights)
            new_weights = self.comm.recv_master()
            self.ml.set_weights(new_weights)
            stop = self.comm.recv_master()
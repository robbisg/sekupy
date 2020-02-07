import numpy as np
import itertools


class ConnectivityStateSimulator():

    def __init__(self, n_nodes=10, max_edges=5, fsamp=256):
    

        edges = [e for e in itertools.combinations(np.arange(n_nodes), 2)]
        n_edges = len(edges)

        states = []
        for i in range(max_edges):
            states += [e for e in itertools.combinations(np.arange(n_edges), i+1)]

        states = np.array(states)

        self._edges = edges
        self._n_edges = len(edges)
        self._states = states
        self._n_states = len(states)

        self._n_nodes = n_nodes
        self._max_edges = max_edges
        self._fs = fsamp



    def fit(self, n_brain_states=6, length_states=100, min_time=2.5, max_time=3.5):

        # Randomize random stuff

        states_idx = np.random.randint(0, self._n_states, n_brain_states)        
        
        selected_states = self._states[states_idx]

        length_bs = np.random.randint(self._fs * min_time, 
                                      self._fs * max_time, 
                                      length_states
                                      )

        # This is done using Hidden Markov Models but since 
        # Transition matrix is uniformly distributed we can use random sequency
        #     mc = mcmix(nBS,'Fix',ones(nBS)*(1/nBS));
        #     seqBS= simulate(mc,nseqBS-1);
        seqBS = np.random.randint(0, n_brain_states, length_states)

        return lenght_bs, sequence_bs

        full_bs_matrix = np.zeros((n_brain_states, n_nodes, n_nodes, model.order))

        for j in range(n_brain_states):
            matrix = np.eye(n_nodes)
            selected_edges = selected_states[j]
            for edge in selected_edges:
                matrix[edges[np.array(edge)]] = 1
            pl.figure()
            pl.imshow(matrix)
            matrix = np.dstack([matrix for _ in range(model.order)])

            cycles = 0
            FA = np.zeros((n_nodes*model.order, n_nodes*model.order))
            eye_idx = n_nodes*model.order - n_nodes
            FA[n_nodes:, :eye_idx] = np.eye(eye_idx)
            for k in range(10000):
                A = 1/2.5 * np.random.rand(n_nodes, n_nodes, model.order) * matrix
                FA[:n_nodes,:] = np.reshape(A, (n_nodes, -1), 'F')

                eig, _ = sp.linalg.eig(FA)

                if np.all(np.abs(eig) < 1):
                    print(k)
                    break

            
            if k==10000:
                raise("Solutions not found")



            full_bs_matrix[j] = A.copy()

        data = model.fit(full_bs_matrix, seqBS, length_bs)
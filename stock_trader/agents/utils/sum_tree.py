'''
The sum tree data structure for prioritized replay.
'''
import numpy as np

class SumTree:
    '''
    The sum tree data structure for prioritized replay.
    '''
    def __init__(self, buffer_size: int) -> None:
        '''
        Initialize the sum tree.

        Parameters
        ----------
        buffer_size : int
            The size of the replay buffer.
        '''
        self.buffer_size = buffer_size
        self.tree = np.zeros(2 * buffer_size - 1)
        self.buffer = np.zeros(buffer_size, dtype=object)
        self.position = 0

    def add(self, priority: float, sample: tuple) -> None:
        '''
        Add a sample to the sum tree.

        Parameters
        ----------
        priority : float
            The priority of the sample.
        sample : tuple
            The sample to add to the sum tree.
        '''
        tree_index = self.position + self.buffer_size - 1
        self.buffer[self.position] = sample
        self.update(tree_index, priority)
        self.position = (self.position + 1) % self.buffer_size

    def update(self, tree_index: int, priority: float) -> None:
        '''
        Update the sum tree.

        Parameters
        ----------
        tree_index : int
            The index of the tree.
        priority : float
            The priority of the sample.
        '''
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, value: float) -> tuple:
        '''
        Get the leaf of the sum tree.

        Parameters
        ----------
        value : float
            The value to get the leaf for.

        Returns
        -------
        tuple
            The leaf of the sum tree.
        int
            The index of the leaf.
        '''
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            if value <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                value -= self.tree[left_child_index]
                parent_index = right_child_index

        data_index = leaf_index - self.buffer_size + 1

        return leaf_index, self.tree[leaf_index], self.buffer[data_index]
    
    def total_priority(self) -> float:
        '''
        Get the total priority of the sum tree.

        Returns
        -------
        float
            The total priority of the sum tree.
        '''
        return self.tree[0]
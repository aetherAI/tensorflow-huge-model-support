from tensorflow_huge_model_support.utils import USE_TF_2, get_gpu_mem_size

import networkx as nx
import tensorflow as tf
import os
import sys
import pkgutil

MERGE_TYPES = ['Merge']
IDENTITY_LIKE_TYPES = ['Identity', 'Reshape']
SIZE_STATIC_TYPES = ['Identity', 'Reshape', 'Transpose']

def isOpExcluded(op, strict, default_batch_size=None, thres=None):
    if 'HMS/' in op.name:
        return True
    if not strict:
        return False
    if any([arg.is_ref for arg in op.op_def.output_arg]):
        return True
    if any([tensor.dtype == tf.resource for tensor in op.outputs]):
        return True

    if thres != None:
        op_size = 0
        for tensor in op.outputs:
            tensor_size = _EstimateTensorSizeSingleton()(tensor, default_batch_size)
            tensor_size = 0 if tensor_size == None else tensor_size
            op_size += tensor_size
        if op_size < thres:
            return True

    return False

class _EstimateTensorSizeSingleton(object):

    _instance = None

    def __new__(cls, *args, **kw):
        if not cls._instance:
            cls._instance = super(_EstimateTensorSizeSingleton, cls).__new__(cls, *args, **kw)
            cls._instance.cache = {}
            cls._instance.cache_nodtype = {}

        return cls._instance

    def __call__(self, tensor, default_batch_size=None):
        if tensor in self.cache:
            return self.cache[tensor]

        if tensor.op.type in IDENTITY_LIKE_TYPES:
            return 0

        if tensor.dtype == tf.float16 or tensor.dtype == tf.int16:
            size = dtype_size = 2
        elif tensor.dtype == tf.float32 or tensor.dtype == tf.int32:
            size = dtype_size = 4
        else:
            size = dtype_size = 8

        def get_tensor_size(tensor, default_batch_size=None):
            if tensor in self.cache_nodtype:
                return self.cache_nodtype[tensor]

            # If the tensor shape is normal
            try:
                tensor_size = 1
                for dim in tensor.shape:
                    tensor_size *= int(dim)
                return tensor_size
            except:
                pass

            # If the tensor is size-static
            if tensor.op.type in SIZE_STATIC_TYPES:
                return get_tensor_size(tensor.op.inputs[0])

            # If the tensor shape is only missing on batch dimension
            tensor_size = 1
            try:
                for idx, dim in enumerate(tensor.shape):
                    if dim.value == None and idx == 0 and default_batch_size != None:
                        dim = default_batch_size
                    tensor_size *= int(dim)
                return tensor_size
            except:
                pass

            # Assume the tensor's size is equal to its largest input
            tensor_size = 0
            for input_tensor in tensor.op.inputs:
                input_tensor_size = get_tensor_size(input_tensor, default_batch_size=default_batch_size)
                tensor_size = max(tensor_size, input_tensor_size) if input_tensor_size != None else tensor_size
            if tensor_size != 0:
                return tensor_size

            # Just give up
            return None
            
        tensor_size = get_tensor_size(tensor, default_batch_size=default_batch_size)
        self.cache_nodtype[tensor] = tensor_size
        size = None if tensor_size == None else int(size * tensor_size)

        self.cache[tensor] = size

        workspace = 0
        if tensor.op.type in ['Conv2D', 'Conv2DBackpropFilter']:
            for input_tensor in tensor.op.inputs:
                input_tensor_size = get_tensor_size(input_tensor, default_batch_size=default_batch_size)
                workspace += (input_tensor_size * dtype_size) if input_tensor_size != None else 0
        elif tensor.op.type in ['Conv2DBackpropInput']:
            for input_tensor in tensor.op.inputs:
                input_tensor_size = get_tensor_size(input_tensor, default_batch_size=default_batch_size)
                workspace += (input_tensor_size * dtype_size) if input_tensor_size != None else 0
            workspace += size if size != None else 0
        size = None if size == None else size + workspace

        return size

class _DependencyHelper(object):
    SCOPE_GROUP = 'group' # the entire group
    SCOPE_CRITICAL = 'critical' # the critical path in the group
    SCOPE_TARGET_OP = 'target_op' # e.g. train_step, loss
    SCOPE_OP = 'op' # A single operation

    def __init__(self, op_graph, target_op, critical_path):
        self._op_graph = op_graph
        self._target_op = target_op
        self._critical_path = critical_path

    def add_dep(self, src, src_scope, dst, dst_scope):
        # Check arguments
        def check_arguments(target, scope):
            if scope in [self.SCOPE_GROUP, self.SCOPE_CRITICAL]:
                assert(isinstance(target, _OperationGraph))
            elif scope in [self.SCOPE_TARGET_OP]:
                assert(target == None)
            elif scope in [self.SCOPE_OP]:
                assert(isinstance(target, tf.Operation))
            else:
                assert False

        check_arguments(src, src_scope)
        check_arguments(dst, dst_scope)

        # Build dependencies
        def get_op_list(target, scope, is_src):
            if scope == self.SCOPE_GROUP:
                if is_src:
                    return target.get_exit_nodes()
                else:
                    return target.get_entry_nodes()
            elif scope == self.SCOPE_CRITICAL:
                if is_src:
                    return target.get_exit_nodes(on_path=self._critical_path)
                else:
                    return target.get_entry_nodes(on_path=self._critical_path)
            elif scope == self.SCOPE_TARGET_OP:
                return [self._target_op]
            elif scope == self.SCOPE_OP:
                return [target]
            else:
                assert False

        aggregation = tf.group(get_op_list(src, src_scope, is_src=True))
        for dst_op in get_op_list(dst, dst_scope, is_src=False):
            dst_op._add_control_inputs([aggregation])

class _OperationGraph(object):
    '''
    Graph of ops and their computational dependencies.
    '''
    def __init__(
        self,
        nx_graph,
        seen_ops,
        default_batch_size=None,
        thres=None
    ):
        '''
        Create a op graph.

        Args:
            nx_graph: An nx.Graph() object.
        '''
        self._nx_graph = nx_graph
        self._seen_ops = seen_ops
        self._default_batch_size = default_batch_size
        self._thres = thres

    def __str__(self):
        return str(list(nx.algorithms.dag.topological_sort(self._nx_graph)))

    def get_critical_path(self):
        '''
        Get the longest path in the graph. This is used for grouping.
        '''
        critical_path = nx.algorithms.dag.dag_longest_path(self._nx_graph)

        # Remove operations right behind identity operations
        new_critical_path = []
        is_identity = False
        for op in reversed(critical_path):
            last_is_identity = is_identity
            if op.type in IDENTITY_LIKE_TYPES:
                is_identity = True
            else:
                is_identity = False
            if not last_is_identity:
                new_critical_path.append(op)
        new_critical_path = list(reversed(new_critical_path))

        return new_critical_path

    def get_entry_nodes(self, on_path=None):
        nx_graph_wo_seen_ops = self._nx_graph.copy()
        nx_graph_wo_seen_ops.remove_nodes_from(self._seen_ops)
        if on_path != None:
            nx_graph_wo_seen_ops.remove_nodes_from(list(
                set(nx_graph_wo_seen_ops.nodes()) - 
                set(on_path)
            ))

        exclude_nodes = set()
        for node in nx_graph_wo_seen_ops.nodes():
            if isOpExcluded(node, strict=True, default_batch_size=self._default_batch_size, thres=self._thres):
                exclude_nodes.add(node)

        while True:
            entry_nodes = set()
            for node, in_degree in nx_graph_wo_seen_ops.in_degree:
                if in_degree == 0:
                    entry_nodes.add(node)
            remove_nodes = list(entry_nodes & exclude_nodes)
            if remove_nodes == []:
                break
            for node in remove_nodes:
                nx_graph_wo_seen_ops.remove_node(node)

        return list(entry_nodes)
        
    def get_exit_nodes(self, on_path=None):
        nx_graph = self._nx_graph.copy()
        if on_path != None:
            nx_graph.remove_nodes_from(list(
                set(nx_graph.nodes()) - 
                set(on_path)
            ))

        exclude_nodes = set()
        for node in nx_graph.nodes():
            if isOpExcluded(node, strict=True, default_batch_size=self._default_batch_size, thres=self._thres):
                exclude_nodes.add(node)

        while True:
            exit_nodes = set()
            for node, out_degree in nx_graph.out_degree:
                if out_degree == 0:
                    exit_nodes.add(node)
            remove_nodes = list(exit_nodes & exclude_nodes)
            if remove_nodes == []:
                break
            for node in remove_nodes:
                nx_graph.remove_node(node)

        return list(exit_nodes)

    def get_ops(self):
        return list(self._nx_graph.nodes)

    def get_unseen_ops(self):
        return list(set(self._nx_graph.nodes) - set(self._seen_ops))

    def _check_size(self, tensor, thres):
        if thres == 0:
            return True
        size = _EstimateTensorSizeSingleton()(tensor, self._default_batch_size)
        if size == None:
            return False
        if size >= thres:
            return True
        else:
            return False

    def get_tensors(self, thres=0):
        tensor_list = []
        for op in self.get_ops():
            for tensor in op.outputs:
                if not self._check_size(tensor, thres):
                    continue
                tensor_list.append(tensor)

        return tensor_list

    def get_seen_tensors(self, thres=0):
        tensor_list = []
        for op in self._seen_ops:
            for tensor in op.outputs:
                if not self._check_size(tensor, thres):
                    continue
                tensor_list.append(tensor)

        return tensor_list

    def get_unseen_tensors(self, thres=0):
        return list(
            set(self.get_tensors(thres)) - 
            set(self.get_seen_tensors(thres))
        )

    def estimate_working_set_size(self):
        result = 0
        for op in self.get_unseen_ops():
            if 'implied_size' in self._nx_graph.node[op]:
                result += self._nx_graph.node[op]['implied_size']
        for tensor in self.get_tensors():
            size = _EstimateTensorSizeSingleton()(tensor, self._default_batch_size)
            result += size if size != None else 0

        return result

    def warmup_tensor_size_estimator(self):
        for op in nx.algorithms.dag.topological_sort(self._nx_graph):
            for tensor in op.outputs:
                _EstimateTensorSizeSingleton()(tensor, self._default_batch_size)

    def get_serialized_unseen_ops(self):
        # Use DFS to do topological sort. Using DFS minimizes the number of new control dependency edges.
        serialized = []
        new_control_deps = []
        stack = self.get_entry_nodes()
        unseen_ops = self.get_unseen_ops()
        done_ops = list(self._seen_ops)
        if len(stack) != 0:
            node = stack.pop()
            serialized.append(node)
            done_ops.append(node)
            is_branch_end = True
            for next_node in self._nx_graph.successors(node):
                if next_node not in unseen_ops:
                    continue
                if next_node in done_ops:
                    continue
                if not all([pred in done_ops for pred in self._nx_graph.predecessors(next_node)]):
                    continue
                stack.append(next_node)
                is_branch_end = False
            if is_branch_end and len(stack) > 0:
                new_control_deps.append((node, stack[-1]))

        return serialized, new_control_deps

    def has_dependency(self, src, dst):
        return (dst in self._nx_graph.successors(src) )

    def __contains__(self, input):
        if isinstance(input, tf.Tensor):
            return input in self.get_tensors()
        elif isinstance(input, tf.Operation):
            return input in self.get_ops()
        else:
            raise

    def summary(self, verbose=False):
        def simplify(op_list):
            return [op.name for op in op_list]

        if verbose:
            string = 'Graph Summary: \n'
            string += 'New nodes: {}\n'.format(simplify(self.get_unseen_ops()))
            string += 'Seen nodes: {}\n'.format(simplify(set(self.get_ops()) - set(self.get_unseen_ops())))
            string += 'Entry nodes: {}\n'.format(simplify(self.get_entry_nodes()))
            string += 'Exit nodes: {}\n'.format(simplify(self.get_exit_nodes()))
        else:
            string = 'Exit nodes: {}'.format(simplify(self.get_exit_nodes()))

        return string

    @staticmethod
    def get_graph_by_target(target_op, seen_ops=[], default_batch_size=None, thres=None):
        '''
        Build a op graph associated to computing a target op.

        Args:
            target_op: A tf.Operation.
        '''
        # Convert the Tensorflow graph to a Networkx graph,
        nx_graph = nx.DiGraph()
        seen_ops_used = set()

        pending_set = set([target_op])
        visited_set = set([target_op])
        while len(pending_set) != 0:
            op = pending_set.pop()
            if op not in seen_ops:
                for from_tensor in op.inputs:
                    from_op = from_tensor.op
                    if isOpExcluded(from_op, strict=False):
                        continue
                    nx_graph.add_edge(from_op, op, is_control_dep=False)
                    if from_op not in visited_set:
                        pending_set.add(from_op)
                        visited_set.add(from_op)
                for from_op in op.control_inputs:
                    if isOpExcluded(from_op, strict=False):
                        continue
                    if from_op in seen_ops:
                        continue
                    nx_graph.add_edge(from_op, op, is_control_dep=True)
                    if from_op not in visited_set:
                        pending_set.add(from_op)
                        visited_set.add(from_op)
            else:
                seen_ops_used.add(op)
        
        return _OperationGraph(nx_graph, list(seen_ops_used), default_batch_size, thres)

    @staticmethod
    def trim_conditional_branches(op_graph):
        op_graph.warmup_tensor_size_estimator()

        nx_graph = op_graph._nx_graph.copy()
        seen_ops = set(op_graph._seen_ops)

        # Deal with conditional operations in the graph by bypassing the branching paths.
        dealt_merge_ops = set()
        while True:
            graph_modified = False
            for op in reversed(list(nx.algorithms.dag.topological_sort(nx_graph))):
                # Skip the op if it's non-Merge op, seen or dealt with.
                if op.type not in MERGE_TYPES:
                    continue
                if op in seen_ops:
                    continue
                if op in dealt_merge_ops:
                    continue

                # Find nondeterministic_ops and last_deterministic_ops
                nondeterministic_ops = set()
                last_deterministic_ops = set()
                nondeterministic_ops_size = 0

                for switch in nx_graph.predecessors(op):
                    pending_list = [switch]
                    local_nondeterministic_ops = set()
                    local_last_deterministic_ops = set()

                    while len(pending_list) != 0:
                        candidate_op = pending_list[0]
                        pending_list = pending_list[1: ]
                        if all([succ in local_nondeterministic_ops | set([op]) for succ in nx_graph.successors(candidate_op)]):
                            local_nondeterministic_ops.add(candidate_op)
                            pending_list += nx_graph.predecessors(candidate_op)
                        else:
                            last_deterministic_ops.add(candidate_op)

                    local_nondeterministic_ops_size = 0
                    for nd_op in local_nondeterministic_ops:
                        if 'implied_size' in nx_graph[nd_op]:
                            local_nondeterministic_ops_size += nx_graph[nd_op]['implied_size']
                        for tensor in nd_op.outputs:
                            size = _EstimateTensorSizeSingleton()(tensor, op_graph._default_batch_size)
                            local_nondeterministic_ops_size += size if size != None else 0

                    nondeterministic_ops |= local_nondeterministic_ops
                    last_deterministic_ops |= local_last_deterministic_ops
                    nondeterministic_ops_size = max(nondeterministic_ops_size, local_nondeterministic_ops_size)

                # Remove nondeterministic_ops from the graph
                nx_graph.remove_nodes_from(nondeterministic_ops)
                seen_ops -= nondeterministic_ops

                # Link last_deterministic_ops with the merge op
                for last_deterministic_op in last_deterministic_ops:
                    nx_graph.add_edge(last_deterministic_op, op, is_control_dep=False) # TODO(chenchc): is_control_dep=False OK?

                # Annotate the total size of non-deterministic ops on the merge op
                nx_graph.node[op]['implied_size'] = nondeterministic_ops_size

                dealt_merge_ops.add(op)
                graph_modified = True
                break

            if not graph_modified:
                break

        return _OperationGraph(nx_graph, list(seen_ops), op_graph._default_batch_size, op_graph._thres)

    @staticmethod
    def get_subgraph_by_target_extended(op_graph, target_op, critical_ops, seen_ops=[]):
        nx_graph = nx.DiGraph()
        seen_ops_used = set()
        seen_ops_set = set(seen_ops + op_graph._seen_ops)
        critical_ops_set = set(critical_ops)

        # Ancestor nodes of target_op
        pending_set = set([target_op])
        while len(pending_set) != 0:
            op = pending_set.pop()
            if op not in seen_ops_set:
                for in_edge in op_graph._nx_graph.in_edges(op, data=True):
                    from_op = in_edge[0]
                    is_control_dep = in_edge[2]['is_control_dep']
                    if is_control_dep and from_op in seen_ops_set:
                        continue
                    nx_graph.add_edge(from_op, op, is_control_dep=is_control_dep)
                    pending_set.add(from_op)
            else:
                seen_ops_used.add(op)

        # Augment more releated nodes - successor nodes of the subnet that is not on critical path
        pending_set = set()
        for node in nx_graph:
            if node in seen_ops_used:
                continue
            for succ in op_graph._nx_graph.successors(node):
                pending_set.add(succ)

        while len(pending_set) != 0:
            op = pending_set.pop()
            if op in nx_graph:
                continue
            if op in critical_ops_set:
                continue
            if any([
                pred not in nx_graph and 
                pred not in seen_ops_set 
                for pred in op_graph._nx_graph.predecessors(op)
            ]):
                continue
            for in_edge in op_graph._nx_graph.in_edges(op, data=True):
                from_op = in_edge[0]
                is_control_dep = in_edge[2]['is_control_dep']
                if is_control_dep:
                    continue
                if from_op not in nx_graph:
                    assert from_op in seen_ops_set
                    seen_ops_used.add(from_op)
                nx_graph.add_edge(from_op, op, is_control_dep=is_control_dep)

            pending_set.update(set(op_graph._nx_graph.successors(op)))

        return _OperationGraph(nx_graph, list(seen_ops_used), op_graph._default_batch_size, op_graph._thres)

class _AddrAgent(object):
    def __init__(self, pf_module):
        self._pf_module = pf_module

        self._tensor_list = []
        self._addr_var_dict = {}
        self._addr_var_updated_dict = {}
        self._addr_update_dict = {}

    def __contains__(self, tensor):
        return tensor in self._tensor_list

    def add_tensor(self, tensor):
        name_prefix = 'HMS/AddrAgent/' + tensor.name.replace(':', '_') + '/'
        if USE_TF_2:
            assign = tf.compat.v1.assign
        else:
            assign = tf.assign

        addr_latest = self._pf_module.get_tensor_addr(tensor, name=(name_prefix + 'addr_latest'))
        with tf.device('/cpu:0'):
            addr_var = tf.Variable(
                initial_value=tf.zeros(shape=[2], dtype=tf.int64),
                name=(name_prefix + 'addr_var'), 
                trainable=False
            )
            addr_var_updated = assign(addr_var, addr_latest, use_locking=False, name=(name_prefix + 'addr_var_updated'))
            addr_update = addr_var_updated.op

        self._tensor_list.append(tensor)
        self._addr_var_dict[tensor] = addr_var
        self._addr_var_updated_dict[tensor] = addr_var_updated
        self._addr_update_dict[tensor] = addr_update

    def get_addr_var(self, tensor):
        return self._addr_var_dict[tensor]

    def get_addr_var_updated(self, tensor):
        return self._addr_var_updated_dict[tensor]

    def get_addr_update(self, tensor):
        return self._addr_update_dict[tensor]

    def __add__(self, rhs):
        def merge_two_dict(dict1, dict2):
            result = dict1.copy()
            result.update(dict2)
            return result

        result = _AddrAgent(self._pf_module)
        result._tensor_list = self._tensor_list + rhs._tensor_list
        result._addr_var_dict = merge_two_dict(self._addr_var_dict, rhs._addr_var_dict)
        result._addr_var_updated_dict = merge_two_dict(self._addr_var_updated_dict, rhs._addr_var_updated_dict)
        result._addr_update_dict = merge_two_dict(self._addr_update_dict, rhs._addr_update_dict)

        return result

    def get_tensors(self):
        return self._tensor_list

class _PrefetchAgent(object):
    id_counter = 0

    def __init__(self, addr_agent, pf_module):
        self._addr_agent = addr_agent
        self._pf_module = pf_module
        self._addr_list = []
        self._id = _PrefetchAgent.id_counter
        _PrefetchAgent.id_counter += 1

    def add_tensors(self, tensors):
        addr_list = []
        for tensor in tensors:
            addr_list.append(self._addr_agent.get_addr_var(tensor))
        
        self._addr_list += addr_list

    def add_seen_tensors(self, tensors):
        addr_list = []
        for tensor in tensors:
            addr_list.append(self._addr_agent.get_addr_var_updated(tensor))
        
        self._addr_list += addr_list

    def get_prefetch_op(self, expand_or_shrink):
        if len(self._addr_list) == 0:
            return tf.no_op()

        name_prefix = 'HMS/PrefetchAgent/' + str(self._id) + '/'
        with tf.device('/cpu:0'):
            addr_list_stacked = tf.stack(self._addr_list, name=(name_prefix + 'addr_list_stacked'))

        prefetch_op = self._pf_module.prefetch(
            addr_list_stacked, 
            expand_or_shrink=expand_or_shrink, 
            name=(name_prefix + 'prefetch_op')
        )

        return prefetch_op

class _EvictAgent(object):
    WARMUP_STEPS = 5
    id_counter = 0

    def __init__(self, addr_agent, pf_module):
        self._addr_agent = addr_agent
        self._pf_module = pf_module
        self._tensor_list = []
        self._addr_list = []
        self._exclude_tensor_list = []
        self._exclude_addr_list = []
        self._id = _EvictAgent.id_counter
        _EvictAgent.id_counter += 1

    def _exclude_tensors(self):
        for tensor in self._exclude_tensor_list:
            while tensor in self._tensor_list:
                idx = self._tensor_list.index(tensor)
                self._tensor_list.pop(idx)
                self._addr_list.pop(idx)

    def add_tensors(self, tensors, use_addr_var_updated=False):
        for tensor in tensors:
            self._tensor_list.append(tensor)
            if use_addr_var_updated:
                self._addr_list.append(self._addr_agent.get_addr_var_updated(tensor))
            else:
                self._addr_list.append(self._addr_agent.get_addr_var(tensor))
        
        self._exclude_tensors()

    def add_exclude_tensors(self, tensors, use_addr_var_updated=False):
        for tensor in tensors:
            self._exclude_tensor_list.append(tensor)
            if use_addr_var_updated:
                self._exclude_addr_list.append(self._addr_agent.get_addr_var_updated(tensor))
            else:
                self._exclude_addr_list.append(self._addr_agent.get_addr_var(tensor))
        
        self._exclude_tensors()

    def get_evict_op(self):
        if len(self._addr_list) == 0:
            return tf.no_op()

        if USE_TF_2:
            assign = tf.compat.v1.assign
        else:
            assign = tf.assign

        name_prefix = 'HMS/EvictAgent/' + str(self._id) + '/'
        with tf.device('/cpu:0'):
            warmup_counter = tf.Variable(
                initial_value=tf.zeros(shape=[], dtype=tf.int64),
                name=(name_prefix + 'warmup_counter'), 
                trainable=False
            )
            updated_warmup_counter = assign(warmup_counter, warmup_counter + 1)
            addr_list_stacked = tf.stack(self._addr_list, name=(name_prefix + 'addr_list_stacked'))
            if self._exclude_addr_list != []:
                exclude_addr_list_stacked = tf.stack(self._exclude_addr_list, name=(name_prefix + 'exclude_addr_list_stacked'))
            else:
                exclude_addr_list_stacked = tf.zeros(shape=[1, 2], dtype=tf.int64, name=(name_prefix + 'exclude_addr_list_stacked'))
        evict_op = tf.cond(
            updated_warmup_counter > _EvictAgent.WARMUP_STEPS,
            lambda: self._pf_module.evict(addr_list_stacked, exclude_addr_list_stacked, name=(name_prefix + 'evict_op')),
            lambda: tf.no_op()
        ).op

        return evict_op

class HMS(object):
    '''
    HMS class for Huge Model Support 
    '''
    def __init__(
        self,
        predict_op,
        train_step_op,
        default_batch_size=None,
        hms_pf_module_path=None,
        graph=None,
        fwd_pf_en=True,
        bwd_pf_en=True,
        fwd_pf_seen_only=False, 
        bwd_pf_seen_only=False, # TODO: test if this is better to turn on or not.
        addr_fetch_depth=1,
        pf_thres_size=(1024 * 1024), 
        pf_depth=2,
        fwd_group_exe_en=True,
        bwd_group_exe_en=True,
        group_exe_depth=1,
        fwd_serialize_en=False,
        bwd_serialize_en=False,
        fwd_evict_en=False,
        bwd_evict_en=False,
        evict_depth=1,
        group_maxsize=None,
        gpu_device='/gpu:0',
        hvd=None
    ):
        '''
        Create and initialize an HMS object.

        Args:
            predict_op: A tf.Operation. The final op of the forward pass.
            train_step_op: A tf.Operation. The op used to update the model.
            hms_pf_module_path: A string. The path of the .so file for prefetch.
            graph: A tf.Graph. The graph where the model is placed.
            fwd_pf_en: A bool. Enabling prefetch in forward pass.
            bwd_pf_en: A bool. Enabling prefetch in backward pass.
            pf_depth: An int. The number of groups prefetched in advance.
            fwd_group_exe_en: A bool. Enabling group execution on forward pass.
            bwd_group_exe_en: A bool. Enabling group execution on backward pass.
            group_maxsize: An int. The maximal byte count of a group. Set None to let HMS determine by system limit. 
                           Set 0 to disable grouping.
            gpu_device: A string. The GPU device name.
        '''
        # Validate inputs and set default values
        if graph == None:
            graph = predict_op.graph
        if group_maxsize == None:
            group_maxsize = int(get_gpu_mem_size(hvd=hvd) * 0.25)
        if hms_pf_module_path == None:
            prefetch_package = pkgutil.get_loader('tensorflow_huge_model_support.prefetch')
            hms_pf_module_path = prefetch_package.path

        # Declare data members from function inputs
        self._predict_op = predict_op
        self._train_step_op = train_step_op
        self._pf_module = tf.load_op_library(hms_pf_module_path)
        self._default_batch_size = default_batch_size
        self._graph = graph
        self._fwd_pf_en = fwd_pf_en
        self._bwd_pf_en = bwd_pf_en
        self._fwd_pf_seen_only = fwd_pf_seen_only
        self._bwd_pf_seen_only = bwd_pf_seen_only
        self._addr_fetch_depth = addr_fetch_depth
        self._pf_thres_size = pf_thres_size
        self._pf_depth = pf_depth
        self._fwd_group_exe_en = fwd_group_exe_en
        self._bwd_group_exe_en = bwd_group_exe_en
        self._group_exe_depth = group_exe_depth
        self._fwd_serialize_en = fwd_serialize_en
        self._bwd_serialize_en = bwd_serialize_en
        self._fwd_evict_en = fwd_evict_en
        self._bwd_evict_en = bwd_evict_en
        self._evict_depth = evict_depth
        self._group_maxsize = group_maxsize
        self._gpu_device = gpu_device
        self._hvd = hvd

        self._fwd_addr_agent = None
        self._bwd_addr_agent = None

        # Initialization

    def _grouping(self, op_graph, critical_path, group_maxsize):
        def log(idx, new_group):
            self._log_info(
                'New group {} with {} nodes.'.format(idx, len(new_group.get_ops()))
            )
            self._log_info(new_group.summary())

        seen_ops = set()
        group_list = []
        pending_sub_graph = None
        i = 0
        while i != len(critical_path):
            op = critical_path[i]
            sub_graph = _OperationGraph.get_subgraph_by_target_extended(op_graph, op, critical_path, list(seen_ops))
            working_set_size = sub_graph.estimate_working_set_size()
            if working_set_size < group_maxsize:
                pending_sub_graph = sub_graph
                i += 1
            elif pending_sub_graph == None:
                group_list.append(sub_graph)
                log(len(group_list) - 1, sub_graph)
                seen_ops.update(sub_graph.get_ops())
                i += 1
            else:
                group_list.append(pending_sub_graph)
                log(len(group_list) - 1, pending_sub_graph)
                seen_ops.update(pending_sub_graph.get_ops())
                pending_sub_graph = None

        if pending_sub_graph != None:
            group_list.append(pending_sub_graph)
            log(len(group_list) - 1, pending_sub_graph)
            seen_ops.update(pending_sub_graph.get_ops())
            pending_sub_graph = None

        assert seen_ops == set(op_graph.get_ops())

        return group_list

    def _do_serialize(self, group_list):
        for i, group in enumerate(group_list):
            serialized_ops, new_control_deps = group.get_serialized_unseen_ops()
            for dep in new_control_deps:
                dep[1]._add_control_inputs([dep[0]])

    def _do_group_execution(self, group_list, dep_helper, prev_group_list=None):
        for i, group in enumerate(group_list):
            if i >= self._group_exe_depth + 1:
                dep_helper.add_dep(
                    src=(group_list[i - self._group_exe_depth - 1]),
                    src_scope=dep_helper.SCOPE_GROUP,
                    dst=group,
                    dst_scope=dep_helper.SCOPE_GROUP
                )
            elif prev_group_list != None:
                dep_helper.add_dep(
                    src=(prev_group_list[len(prev_group_list) + i - self._group_exe_depth - 1]),
                    src_scope=dep_helper.SCOPE_GROUP,
                    dst=group,
                    dst_scope=dep_helper.SCOPE_GROUP
                )

            #self._log_info('Group execution built on group {}.'.format(i))

    def _do_addr_fetch(self, target_op, group_list, addr_agent, critical_path, dep_helper, prev_addr_agent=None):
        # Build addr fetching operations and store in addr agent.
        for i, group in enumerate(group_list):
            aggregate_op = tf.no_op()
            for tensor in group.get_tensors(thres=self._pf_thres_size):
                if prev_addr_agent != None and tensor in prev_addr_agent:
                    continue
                if tensor in addr_agent:
                    continue
                if tensor.op == target_op:
                    continue

                addr_agent.add_tensor(tensor)

                dep_helper.add_dep(
                    src=(addr_agent.get_addr_update(tensor)), 
                    src_scope=dep_helper.SCOPE_OP,
                    dst=aggregate_op,
                    dst_scope=dep_helper.SCOPE_OP
                )

            if i + self._addr_fetch_depth + 1 < len(group_list):
                dep_helper.add_dep(
                    src=aggregate_op, 
                    src_scope=dep_helper.SCOPE_OP,
                    dst=(group_list[i + self._addr_fetch_depth + 1]),
                    dst_scope=dep_helper.SCOPE_GROUP
                )
            else:
                dep_helper.add_dep(
                    src=aggregate_op, 
                    src_scope=dep_helper.SCOPE_OP,
                    dst=None,
                    dst_scope=dep_helper.SCOPE_TARGET_OP
                )

            #self._log_info('Addr fetch built on group {}.'.format(i))

    def _do_prefetch(self, target_op, group_list, addr_agent, dep_helper, circular_pf_op_list, prev_addr_agent=None, prev_group_list=None, seen_only=False):
        # Build prefetch operations
        if prev_addr_agent == None:
            merged_addr_agent = addr_agent
        else:
            merged_addr_agent = addr_agent + prev_addr_agent

        prev_prefetch_op = None
        for i, group in enumerate(group_list):
            pf_agent = _PrefetchAgent(merged_addr_agent, self._pf_module)
            if not seen_only:
                pf_agent.add_tensors(list(
                    set(group.get_unseen_tensors(thres=self._pf_thres_size)) - 
                    set(target_op.outputs)
                ))
            for tensor in list(
                set(group.get_seen_tensors(thres=self._pf_thres_size)) - 
                set(target_op.outputs)
            ):
                recent_access = False
                for j in range(1, self._addr_fetch_depth + 1):
                    if i - j >= 0:
                        recent_access = recent_access or (tensor in group_list[i - j].get_unseen_tensors())
                    elif prev_group_list != None:
                        recent_access = recent_access or (tensor in prev_group_list[len(prev_group_list) + i - j].get_unseen_tensors())
                if recent_access:
                    #pf_agent.add_tensors([tensor]) # TODO: Check if this is necessary.
                    pass
                else:
                    pf_agent.add_seen_tensors([tensor])

            prefetch_op = pf_agent.get_prefetch_op(True)
            
            if i >= self._pf_depth + 1:
                dep_helper.add_dep(
                    src=(group_list[i - self._pf_depth - 1]),
                    src_scope=dep_helper.SCOPE_GROUP,
                    dst=prefetch_op,
                    dst_scope=dep_helper.SCOPE_OP
                )
                dep_helper.add_dep(
                    src=prefetch_op,
                    src_scope=dep_helper.SCOPE_OP,
                    dst=group,
                    dst_scope=dep_helper.SCOPE_GROUP
                )
            elif prev_group_list != None:
                dep_helper.add_dep(
                    src=(prev_group_list[len(prev_group_list) + i - self._pf_depth - 1]),
                    src_scope=dep_helper.SCOPE_GROUP,
                    dst=prefetch_op,
                    dst_scope=dep_helper.SCOPE_OP
                )
                dep_helper.add_dep(
                    src=prefetch_op,
                    src_scope=dep_helper.SCOPE_OP,
                    dst=group,
                    dst_scope=dep_helper.SCOPE_GROUP
                )
            else:
                circular_pf_op_list.append(prefetch_op)

            prev_prefetch_op = prefetch_op
            #self._log_info('Prefetch built on group {}.'.format(i))

        if prev_group_list != None:
            assert len(circular_pf_op_list) == self._pf_depth + 1
            for i in range(self._pf_depth + 1):
                dep_helper.add_dep(
                    src=(group_list[len(group_list) - i - self._pf_depth - 1]),
                    src_scope=dep_helper.SCOPE_GROUP,
                    dst=circular_pf_op_list[i],
                    dst_scope=dep_helper.SCOPE_OP
                )
                dep_helper.add_dep(
                    src=circular_pf_op_list[i],
                    src_scope=dep_helper.SCOPE_OP,
                    dst=None,
                    dst_scope=dep_helper.SCOPE_TARGET_OP
                )

    def _modify_graph(
        self, 
        target_op, 
        addr_agent, 
        circular_pf_op_list,
        seen_ops=[], 
        prefetch_en=True, 
        evict_en=False,
        prefetch_seen_only=False, 
        group_exe_en=True, 
        serialize_en=False,
        prev_addr_agent=None, 
        prev_group_list=None
    ):
        '''
        Modify graph on either forward pass or backward pass.

        Args:
            target_op: A Operation. predict_op for forward pass and train_step_op for backward pass.
            seen_ops: A list of Operations. Operations that have been already computed.
            prefetch_en: A bool. Enable prefetch or not.
            group_exe_en: A bool. Enable group execution or not.
        '''
        op_graph = _OperationGraph.get_graph_by_target(target_op, seen_ops=seen_ops, default_batch_size=self._default_batch_size, thres=self._pf_thres_size)
        op_graph = _OperationGraph.trim_conditional_branches(op_graph)
        critical_path = op_graph.get_critical_path()
        dep_helper = _DependencyHelper(op_graph, target_op, critical_path)
        group_list = self._grouping(op_graph, critical_path, self._group_maxsize)
        if serialize_en:
            self._log_info('Serializing.')
            self._do_serialize(group_list)
        elif group_exe_en:
            self._log_info('Building group execution.')
            self._do_group_execution(group_list, dep_helper, prev_group_list=prev_group_list)
        if prefetch_en or evict_en:
            self._log_info('Building address fetch.')
            self._do_addr_fetch(target_op, group_list, addr_agent, critical_path, dep_helper, prev_addr_agent=prev_addr_agent)
        if prefetch_en:
            self._log_info('Building prefetching.')
            self._do_prefetch(
                target_op, 
                group_list, 
                addr_agent, 
                dep_helper,
                circular_pf_op_list,
                prev_addr_agent=prev_addr_agent, 
                prev_group_list=prev_group_list, 
                seen_only=prefetch_seen_only
            )

        return group_list, dep_helper

    def _do_evict(self, target_op, group_list, dep_helper, merged_addr_agent, pf_en, next_group_list=None):
        # Build evict operations
        for i, group in enumerate(group_list):
            if next_group_list == None and i + self._evict_depth + 1 >= len(group_list):
                # One of the very last groups. Skip eviction.
                # TODO: Is circular eviction helpful?
                continue

            evict_agent = _EvictAgent(merged_addr_agent, self._pf_module)
            evict_agent.add_tensors(
                group.get_tensors(thres=self._pf_thres_size),
                use_addr_var_updated=(self._evict_depth > self._addr_fetch_depth)
            )

            exclude_depth = self._evict_depth
            if pf_en:
                exclude_depth = max(exclude_depth, self._pf_depth + 1)
            for dist in range(1, exclude_depth + 1):
                if i + dist < len(group_list):
                    evict_agent.add_exclude_tensors(
                        group_list[i + dist].get_tensors(thres=self._pf_thres_size),
                        use_addr_var_updated=(self._evict_depth - dist > self._addr_fetch_depth)
                    )
                elif next_group_list != None and i + dist - len(group_list) < len(next_group_list):
                    evict_agent.add_exclude_tensors(
                        next_group_list[i + dist - len(group_list)].get_tensors(thres=self._pf_thres_size),
                        use_addr_var_updated=(self._evict_depth - dist > self._addr_fetch_depth)    
                    )

            evict_op = evict_agent.get_evict_op()

            if i + self._evict_depth + 1 < len(group_list):
                dep_helper.add_dep(
                    src=group,
                    src_scope=dep_helper.SCOPE_GROUP,
                    dst=evict_op,
                    dst_scope=dep_helper.SCOPE_OP
                )
                dep_helper.add_dep(
                    src=evict_op,
                    src_scope=dep_helper.SCOPE_OP,
                    dst=(group_list[i + self._evict_depth + 1]),
                    dst_scope=dep_helper.SCOPE_GROUP
                )
            elif next_group_list != None:
                dep_helper.add_dep(
                    src=group,
                    src_scope=dep_helper.SCOPE_GROUP,
                    dst=evict_op,
                    dst_scope=dep_helper.SCOPE_OP
                )
                dep_helper.add_dep(
                    src=evict_op,
                    src_scope=dep_helper.SCOPE_OP,
                    dst=(next_group_list[i + self._evict_depth + 1 - len(group_list)]),
                    dst_scope=dep_helper.SCOPE_GROUP
                )
            else:
                raise

    def run(self):
        '''
        Start modifying the graph to invoke HMS.
        '''
        self._log_info('Start modifying the graph.')

        # Modify graph on forward pass
        circular_pf_op_list = list()
        fwd_ops = None
        if self._fwd_group_exe_en or self._fwd_pf_en or self._fwd_evict_en:
            self._log_info('Processing on forward pass.')
            self._fwd_addr_agent = _AddrAgent(self._pf_module)
            fwd_group_list, fwd_dep_helper = self._modify_graph(
                self._predict_op,
                self._fwd_addr_agent,
                circular_pf_op_list,
                seen_ops=[],
                prefetch_en=self._fwd_pf_en,
                evict_en=self._fwd_evict_en,
                prefetch_seen_only=self._fwd_pf_seen_only,
                group_exe_en=self._fwd_group_exe_en,
                serialize_en=self._fwd_serialize_en
            )

        # Modify graph on backward pass
        if self._bwd_group_exe_en or self._bwd_pf_en or self._bwd_evict_en:
            self._log_info('Processing on backward pass.')
            seen_ops = _OperationGraph.get_graph_by_target(self._predict_op, default_batch_size=self._default_batch_size).get_ops()
            self._bwd_addr_agent = _AddrAgent(self._pf_module)
            bwd_group_list, bwd_dep_helper = self._modify_graph(
                self._train_step_op,
                self._bwd_addr_agent,
                circular_pf_op_list,
                seen_ops=seen_ops,
                prefetch_en=self._bwd_pf_en,
                evict_en=self._bwd_evict_en,
                prefetch_seen_only=self._bwd_pf_seen_only,
                group_exe_en=self._bwd_group_exe_en,
                serialize_en=self._bwd_serialize_en,
                prev_addr_agent=self._fwd_addr_agent,
                prev_group_list=fwd_group_list
            )

        # Modify graph on forward pass for eviction
        if self._fwd_evict_en:
            self._log_info('Building eviction on forward pass.')
            self._do_evict(
                self._predict_op,
                fwd_group_list,
                fwd_dep_helper,
                self._fwd_addr_agent + self._bwd_addr_agent,
                self._fwd_pf_en,
                next_group_list=bwd_group_list
            )

        # Modify graph on backward pass for eviction
        if self._bwd_evict_en:
            self._log_info('Building eviction on backward pass.')
            self._do_evict(
                self._train_step_op,
                bwd_group_list,
                bwd_dep_helper,
                self._fwd_addr_agent + self._bwd_addr_agent,
                self._bwd_pf_en,
            )

    def _log_info(self, message):
        '''
        Log debug message.
        '''
        if USE_TF_2:
            tf.get_logger().warn(
                '[HMS] {}'.format(message)
            )
        else:
            tf.logging.log(
                tf.logging.WARN, 
                '[HMS] {}'.format(message)
            )



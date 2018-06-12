'''
Contain a number of useful functions.
'''
import tensorflow as tf

def huber_loss(x,delta=1.0):
    '''
    https://en.wikipedia.org/wiki/Huber_loss
    left part : quadratic
    right part : linear
    '''
    return tf.where(
        tf.abs(x)<delta,
        tf.square(x)*0.5,
        delta*(tf.abs(x)-0.5*delta)
    )
def collect_params():
    pass

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos=device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type=='GPU']
def sample_n_unique(samlling_f,n):
    res=[]
    while len(res)<n:
        candidate=samlling_f()
        if candidate not in res:
            res.append(candidate)
    return res


def initialize_interdependent_variables(session,vars_list,feed_dict):
    '''
    initialize a list of variables,when the initialization of these variables
    depends on the initialization of the other variables.
    '''
    vars_left=vars_list
    while len(vars_left)>0:
        new_vars_left=[]
        for v in vars_left:
            try:
                session.run(tf.variables_initializer([v]),feed_dict)
            # raised when running an operation that reads a tf.Variable before it has been initialized
            except tf.errors.FailedPreconditionError:
                new_vars_left.append(v)
        if len(new_vars_left)>=len(vars_left):
            raise Exception("Cycle in variable dependencies, or extenrnal precondition unsatisfied.")
        else:
            vars_left=new_vars_left
def minimize_and_clip(optimizer,objective,var_list,clip_val=10):
    '''
    Minimize 'objective' using 'optimizer' w.r.t. variables in
    'var_list' while ensure the norm of the gradients for each
    variables is clipped to 'clip_val'.
    '''
    gradients=optimizer.compute_gradients(objective,var_list=var_list)
    for i,(grad,var) in enumerate(gradients):
        if grad is not None:
            gradients[i]=(tf.clip_by_norm(grad,clip_val),var)
    return optimizer.apply_gradients(gradients)

def linear_interpolation(l,r,alpha):
    return l+alpha*(r-l)

class PiecewiseSchedule:
    def __init__(self,endpoints,interpolation=linear_interpolation,outside_value=None):
        '''
        :param endpoints: [(int,int)]
        list of pairs (time,value),when t='time',return 'value'
        if t between 'time1' and 'time2',take interpolation between 'value1' and 'value2'
        :param interpolation: {function}
        the interpolation method,we use linear interpolation
        :param outside_value: {float}
        if t is not in the sections in 'endpoints',then return this value
        '''
        idxes=[e[0] for e in endpoints]
        assert idxes==sorted(idxes)
        self._interpolation=interpolation
        self._outside_value=outside_value
        self._endpoints=endpoints
    def value(self,t):
        for (l_t,l),(r_t,r) in zip(self._endpoints[:-1],self._endpoints[1:]):
            if l_t<=t and t<r_t:
                alpha=float(t-l_t)/(r_t-l_t)
                return self._interpolation(l,r,alpha)
        # if t do not in the time sections
        assert self._outside_value is not None
        return self._outside_value


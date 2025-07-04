import numpy as np
from dynamics.path_generator import cubic_spline_interpolation

def data_gen_robot(dt,num_traj,num_snaps, robot, controller):
    
    """
    A function for generating the data for the given robot
    dt = time step
    num_traj = number of trajectories
    num_snap = number of snaps in each trajectory
    robot = robot
    controller = random or model partitioning controller (controller)
    """
    num_states = robot.random_q().shape[0]
    num_inputs = num_states
    
    if robot.name.__contains__("2D"):
        num_states_cart = 2
    else:
        num_states_cart = 3
    
    X = np.zeros((num_traj, num_snaps, 2*num_states))
    tau = np.zeros((num_traj, num_snaps-1, num_states))
    X_end = np.zeros((num_traj, num_snaps, 2*num_states_cart))
    
    
    if controller == 'random':
        for i in range(num_traj):
            X[i,0,:num_states] = (np.pi/2)*(2*np.random.rand(num_states,1) - 1).reshape(num_states,);
            X[i,0,num_states:] = 0.01*(2*np.random.rand(num_states,1) - 1).reshape(num_states,);

            tf = np.array(robot.fkine(X[i,0,:num_states])); #fkine(q)
            X_end[i,0,:num_states_cart] = tf[:num_states_cart,3];
            X_end[i,0,:] = tf[:num_states_cart,3];
            X_end[i,0,num_states_cart:] = np.matmul(robot.jacob0(X[i,0,:num_states])[:num_states_cart,:],X[i,0,num_states:].reshape(-1,))
            for j in range(num_snaps-1):
                tau[i,j,:] = 0.01*(2*np.random.rand(num_states,1) - 1).reshape(num_states,); #input torques


                th_ddot = robot.accel(X[i,j,:num_states], X[i,j,num_states:], tau[i,j,:]) # forward dynamic(q, th, q_dot) 
                X[i,j+1,num_states:] = th_ddot*dt + X[i,j,num_states:];
                X[i,j+1,:num_states] = X[i,j,num_states:]*dt + X[i,j,:num_states];

                tf = np.array(robot.fkine(X[i,j+1,:num_states]));
                X_end[i,j+1,:] = tf[:num_states_cart,3];
                X_end[i,j+1,num_states_cart:] = np.matmul(robot.jacob0(X[i,j+1,:num_states])[:num_states_cart,:],X[i,j+1,num_states:].reshape(-1,))
        
    else:
        for i in range(num_traj):
            num_wp = 6
            # generate path
            t_end = dt*num_snaps; 
            t_wp = np.zeros((num_wp,))
            q_wp = np.zeros((num_states, num_wp))
            for s in range(num_wp):
                if s == 0:
                    q_wp[:,s] = robot.random_q();
                    t_wp[s] = 0;
                    q_wp_dot = 0.00*(2*np.random.rand(num_states,) - 1);
                else:
                    t_wp[s] = s*t_end/num_wp;
                    q_wp[:,s] = q_wp[:,s-1] + q_wp_dot*(t_wp[s]-t_wp[s-1]);
                    q_wp_dot = 0.001*(2*np.random.rand(num_states,) - 1);


            q_traj, qd_traj, qddot_traj = cubic_spline_interpolation(np.transpose(q_wp),t_end,num_snaps-1);

            X[i,0,:num_states] = q_traj[:,0];
            X[i,0,num_states:] = qd_traj[:,0];

            tf = np.array(robot.fkine(X[i,0,:num_states])); #fkine(q)
            X_end[i,0,:num_states_cart:] = tf[:num_states_cart,3];
            X_end[i,0,num_states_cart:] = np.matmul(robot.jacob0(X[i,0,:num_states])[:num_states_cart,:],X[i,0,num_states:].reshape(-1,))

            Kp = 8;
            Kv = 4;
            tau_dash = np.zeros((num_states,num_snaps))
            for j in range(num_snaps-1):
                tau_dash[:,j] = qddot_traj[:,j] + Kv*(qd_traj[:,j] - X[i,j,num_states:]) + Kp*(q_traj[:,j] - X[i,j,:num_states]);
                tau[i,j,:] = robot.rne(X[i,j,:num_states], X[i,j,num_states:], tau_dash[:,j])

                th_ddot = robot.accel(X[i,j,:num_states], X[i,j,num_states:], tau[i,j,:]) # forward dynamic(q, th, q_dot) 
                X[i,j+1,num_states:] = th_ddot*dt + X[i,j,num_states:];
                X[i,j+1,:num_states] = X[i,j,num_states:]*dt + X[i,j,:num_states];

                tf = np.array(robot.fkine(X[i,j+1,:num_states]));
                X_end[i,j+1,:] = tf[:num_states_cart,3];
                X_end[i,j+1,num_states_cart:] = np.matmul(robot.jacob0(X[i,j+1,:num_states])[:num_states_cart,:],X[i,j+1,num_states:].reshape(-1,))


    
    return X_end, X, tau


def data_gen_robot_multi(dt,num_traj,num_snaps, robot, robot_2, robot_pars_changed, controller):
    
    """
    A function for generating the data for the two robot
    dt = time step
    num_traj = number of trajectories
    num_snap = number of snaps in each trajectory
    robot = 1st robot
    robot_2 = 2nd robot
    controller = random or model partitioning controller (controller)
    """
    num_states = robot.random_q().shape[0]
    num_inputs = num_states
    
    if robot.name.__contains__("2D"):
        num_states_cart = 2
    else:
        num_states_cart = 3
    
    X_1 = np.zeros((num_traj, num_snaps, 2*num_states))
    X_2 = np.zeros((num_traj, num_snaps, 2*num_states))
    X_end_1 = np.zeros((num_traj, num_snaps, 2*num_states_cart))
    X_end_2 = np.zeros((num_traj, num_snaps, 2*num_states_cart))
    tau = np.zeros((num_traj, num_snaps-1, num_states))
    
    
    if controller == 'random':
        for i in range(num_traj):
            X_1[i,0,:num_states] = (np.pi/2)*(2*np.random.rand(num_states,1) - 1).reshape(num_states,);
            X_1[i,0,num_states:] = 0.1*(2*np.random.rand(num_states,1) - 1).reshape(num_states,);

            tf_1 = np.array(robot.fkine(X_1[i,0,:num_states])); #fkine(q)
            X_end_1[i,0,:num_states_cart] = tf_1[:num_states_cart,3];
            X_end_1[i,0,num_states_cart:] = np.matmul(robot.jacob0(X_1[i,0,:num_states])[:num_states_cart,:],X_1[i,0,num_states:].reshape(-1,))
            
            X_2[i,0,:num_states] = X_1[i,0,:num_states];
            X_2[i,0,num_states:] = X_1[i,0,num_states:];
            
            tf_2 = np.array(robot_2.fkine(X_2[i,0,:num_states])); #fkine(q)
            X_end_2[i,0,:num_states_cart] = tf_2[:num_states_cart,3];
            X_end_2[i,0,num_states_cart:] = np.matmul(robot.jacob0(X_2[i,0,:num_states])[:num_states_cart,:],X_2[i,0,num_states:].reshape(-1,))
            
            for j in range(num_snaps-1):
                tau[i,j,:] = 10*(2*np.random.rand(num_states,1) - 1).reshape(num_states,); #input torques


                th_ddot_1 = robot.accel(X_1[i,j,:num_states], X_1[i,j,num_states:], tau[i,j,:]) # forward dynamic(q, th, q_dot) 
                X_1[i,j+1,num_states:] = th_ddot_1*dt + X_1[i,j,num_states:];
                X_1[i,j+1,:num_states] = X_1[i,j,num_states:]*dt + X_1[i,j,:num_states];

                tf_1 = np.array(robot.fkine(X_1[i,j+1,:num_states]));
                X_end_1[i,j+1,:num_states_cart] = tf_1[:num_states_cart,3];
                X_end_1[i,j+1,num_states_cart:] = np.matmul(robot.jacob0(X_1[i,j+1,:num_states])[:num_states_cart,:],X_1[i,j+1,num_states:].reshape(-1,))


                if robot_pars_changed ['ext_torque'] and robot_pars_changed ['ext_torque_type'] == 'constant':
                    tau_app = tau[i,j,:]*(1 + 1/robot_pars_changed['SN_input'])
                elif robot_pars_changed ['ext_torque'] and robot_pars_changed ['ext_torque_type'] == 'gaussian':
                    tau_app = np.random.normal(loc = tau[i,j,:],scale = 1/robot_pars_changed['SN_input'])
                elif robot_pars_changed ['ext_torque'] and robot_pars_changed ['ext_torque_type'] == 'random':
                    tau_app = tau[i,j,:] + (2*np.random.rand(tau[i,j,:].shape[0])-1)/robot_pars_changed['SN_input']
                elif robot_pars_changed ['ext_torque'] and robot_pars_changed ['ext_torque_type'] == 'sinosiodal':
                    tau_app = tau[i,j,:] + np.sin(200*np.pi*j*dt)/robot_pars_changed['SN_input']
                    
                else:
                    tau_app = tau[i,j,:]
    
                
                th_ddot_2 = robot_2.accel(X_2[i,j,:num_states], X_2[i,j,num_states:], tau_app) # forward dynamic(q, th, q_dot) 
                X_2[i,j+1,num_states:] = th_ddot_2*dt + X_2[i,j,num_states:];

                if robot_pars_changed ['state_noise'] and robot_pars_changed ['state_noise_type'] == 'constant':
                    X_2[i,j+1,num_states:] = X_2[i,j+1,num_states:]*(1 + 1/robot_pars_changed['SN_state'])
                elif robot_pars_changed ['state_noise'] and robot_pars_changed ['state_noise_type'] == 'gaussian':
                    X_2[i,j+1,num_states:] = np.random.normal(loc = X_2[i,j+1,num_states:],scale = 1/robot_pars_changed['SN_state'])
                elif robot_pars_changed ['state_noise'] and robot_pars_changed ['state_noise_type'] == 'random':
                    X_2[i,j+1,num_states:] = X_2[i,j+1,num_states:] + (2*np.random.rand(X_2[i,j+1,num_states:].shape[0])-1)/robot_pars_changed['SN_input']
                elif robot_pars_changed ['state_noise'] and robot_pars_changed ['state_noise_type'] == 'sinosiodal':
                    X_2[i,j+1,num_states:] = X_2[i,j+1,num_states:] + np.sin(200*np.pi*j*dt)/robot_pars_changed['SN_noise']
                
                X_2[i,j+1,:num_states] = X_2[i,j,num_states:]*dt + X_2[i,j,:num_states];

                tf_2 = np.array(robot_2.fkine(X_2[i,j+1,:num_states]));
                X_end_2[i,j+1,:num_states_cart] = tf_2[:num_states_cart,3];
                X_end_2[i,j+1,num_states_cart:] = np.matmul(robot.jacob0(X_2[i,j+1,:num_states])[:num_states_cart,:],X_2[i,j+1,num_states:].reshape(-1,))
                

        
    else:
        for i in range(num_traj):
            num_wp = 6
            # generate path
            t_end = dt*num_snaps; 
            t_wp = np.zeros((num_wp,))
            q_wp = np.zeros((num_states, num_wp))
            for s in range(num_wp):
                if s == 0:
                    q_wp[:,s] = robot.random_q();
                    t_wp[s] = 0;
                    q_wp_dot = 0.05*(2*np.random.rand(num_states,) - 1);
                else:
                    t_wp[s] = s*t_end/num_wp;
                    q_wp[:,s] = q_wp[:,s-1] + q_wp_dot*(t_wp[s]-t_wp[s-1]);
                    q_wp_dot = 0.05*(2*np.random.rand(num_states,) - 1);


            q_traj, qd_traj, qddot_traj = cubic_spline_interpolation(np.transpose(q_wp),t_end,num_snaps-1);

            X_1[i,0,:num_states] = q_traj[:,0];
            X_1[i,0,num_states:] = qd_traj[:,0];

            tf_1 = np.array(robot.fkine(X_1[i,0,:num_states])); #fkine(q)
            X_end_1[i,0,:num_states_cart] = tf_1[:num_states_cart,3];
            X_end_1[i,0,num_states_cart:] = np.matmul(robot.jacob0(X_1[i,0,:num_states])[:num_states_cart,:],X_1[i,0,num_states:].reshape(-1,))

            X_2[i,0,:num_states] = X_1[i,0,:num_states];
            X_2[i,0,num_states:] = X_1[i,0,num_states:];

            tf_2 = np.array(robot_2.fkine(X_2[i,0,:num_states])); #fkine(q)
            X_end_2[i,0,:num_states_cart] = tf_2[:num_states_cart,3];
            X_end_2[i,0,num_states_cart:] = np.matmul(robot.jacob0(X_2[i,0,:num_states])[:num_states_cart,:],X_2[i,0,num_states:].reshape(-1,))

            Kp = 8;
            Kv = 4;
            tau_dash = np.zeros((num_states,num_snaps))
            for j in range(num_snaps-1):
                tau_dash[:,j] = qddot_traj[:,j] + Kv*(qd_traj[:,j] - X_1[i,j,num_states:]) + Kp*(q_traj[:,j] - X_1[i,j,:num_states]);
                
                tau[i,j,:] = robot.rne(X_1[i,j,:num_states], X_1[i,j,num_states:], tau_dash[:,j])

                th_ddot_1 = robot.accel(X_1[i,j,:num_states], X_1[i,j,num_states:], tau[i,j,:]) # forward dynamic(q, th, q_dot) 
                X_1[i,j+1,num_states:] = th_ddot_1*dt + X_1[i,j,num_states:];
                X_1[i,j+1,:num_states] = X_1[i,j,num_states:]*dt + X_1[i,j,:num_states];

                tf_1 = np.array(robot.fkine(X_1[i,j+1,:num_states]));
                X_end_1[i,j+1,:num_states_cart] = tf_1[:num_states_cart,3];
                X_end_1[i,j+1,num_states_cart:] = np.matmul(robot.jacob0(X_1[i,j+1,:num_states])[:num_states_cart,:],X_1[i,j+1,num_states:].reshape(-1,))

                if robot_pars_changed ['ext_torque'] and robot_pars_changed ['ext_torque_type'] == 'constant':
                    tau_app = tau[i,j,:](1+1/robot_pars_changed['SN_input'])
                elif robot_pars_changed ['ext_torque'] and robot_pars_changed ['ext_torque_type'] == 'gaussian':
                    tau_app = np.random.normal(loc = tau[i,j,:],scale = 1/robot_pars_changed['SN_input'])
                elif robot_pars_changed ['ext_torque'] and robot_pars_changed ['ext_torque_type'] == 'random':
                    tau_app = tau[i,j,:] + (2*np.random.rand(tau[i,j,:].shape[0])-1)/robot_pars_changed['SN_input']
                elif robot_pars_changed ['ext_torque'] and robot_pars_changed ['ext_torque_type'] == 'sinosiodal':
                    tau_app = tau[i,j,:] + np.sin(200*np.pi*j*dt)/robot_pars_changed['SN_input']
                    
                else:
                    tau_app = tau[i,j,:]
    
                
                th_ddot_2 = robot_2.accel(X_2[i,j,:num_states], X_2[i,j,num_states:], tau_app) # forward dynamic(q, th, q_dot) 
                X_2[i,j+1,num_states:] = th_ddot_2*dt + X_2[i,j,num_states:];
                if robot_pars_changed ['state_noise'] and robot_pars_changed ['state_noise_type'] == 'constant':
                    X_2[i,j+1,num_states:] = X_2[i,j+1,num_states:]*(1 + 1/robot_pars_changed['SN_state'])
                elif robot_pars_changed ['state_noise'] and robot_pars_changed ['state_noise_type'] == 'gaussian':
                    X_2[i,j+1,num_states:] = np.random.normal(loc = X_2[i,j+1,num_states:],scale = 1/robot_pars_changed['SN_state'])
                elif robot_pars_changed ['state_noise'] and robot_pars_changed ['state_noise_type'] == 'random':
                    X_2[i,j+1,num_states:] = X_2[i,j+1,num_states:] + (2*np.random.rand(X_2[i,j+1,num_states:].shape[0])-1)/robot_pars_changed['SN_input']
                elif robot_pars_changed ['state_noise'] and robot_pars_changed ['state_noise_type'] == 'sinosiodal':
                    X_2[i,j+1,num_states:] = X_2[i,j+1,num_states:] + np.sin(200*np.pi*j*dt)/robot_pars_changed['SN_noise']
                
                X_2[i,j+1,:num_states] = X_2[i,j,num_states:]*dt + X_2[i,j,:num_states];

                tf_2 = np.array(robot_2.fkine(X_2[i,j+1,:num_states]));
                X_end_2[i,j+1,:num_states_cart] = tf_2[:num_states_cart,3];
                X_end_2[i,j+1,num_states_cart:] = np.matmul(robot.jacob0(X_2[i,j+1,:num_states])[:num_states_cart,:],X_2[i,j+1,num_states:].reshape(-1,))


    
    return X_end_1, X_1, X_end_2, X_2, tau

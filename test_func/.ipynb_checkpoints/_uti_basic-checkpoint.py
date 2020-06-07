#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#############################################################################
import time
now = time.time
import smtplib, ssl
from multiprocessing import Pool
import multiprocessing

n_cores = multiprocessing.cpu_count()
#############################################################################
dash = "--------------------------------------"
DASH = "\n" + "--------------------------------------" + "\n"
Dash = "\n" + dash
dasH = dash + "\n"
#############################################################################
#%% utility funs

def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))
        
def parmap(f, X, nprocs=multiprocessing.cpu_count()-2):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]

def send_email(message = None, email_address = "13300180059@fudan.edu.cn", title = "Your results are ready!",
              receiver_email = "Same"): # py.notify.me@gmail.com
    port = 465  # For SSL
    # Create a secure SSL context
    context = ssl.create_default_context()
    sender_email = email_address # "py.notify.me@gmail.com"
    if receiver_email == "Same":
        receiver_email = email_address
    email_content = message
    
    a = """

    """
    
    message = """\
    Subject: """ + title + a
    message += email_content
    
    with smtplib.SMTP_SSL("mail.fudan.edu.cn", port, context=context) as server: # "smtp.gmail.com"
        server.login(email_address,"w19950722")  #("py.notify.me@gmail.com", "w19950722")
        server.sendmail(sender_email, receiver_email, message)

#############################################################################
def rep_seeds(fun,rep_times):
    """
    non-parallel-version of pool.map
    """
    return list(map(fun, range(rep_times)))

def rep_seeds_print(fun,rep_times,init_seed):
    r = []
    start = now()
    for seed in range(rep_times):
        r.append(fun(seed + init_seed))
        if seed % 25 == 0:
            print(round((seed+1)/rep_times*100,2),"% DONE", round((now() - start)/60,2), "mins" )
    return r
#############################################################################     

def round_list(thelist,dec):
    """
    extend np.round to list
    """
    return [round(a,dec) for a in thelist]

def print_time_cost(seed,total_rep,time):
    print(round((seed+1/total_rep)*100,3),"% DONE, takes", round((time)/60,3)," mins \n")
    
def is_disc(v, n):
    return len(set(v)) <= n

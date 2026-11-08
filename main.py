def real_step(x,u):return .9*x+u-.04*x*x
def learned_step(x,u):return .9*x+u
def evaluate(policy,steps=30):
    real=pred=0.;predicted_return=actual_return=0.;errors=[]
    for t in range(steps):
        u=policy(pred,t);p=learned_step(pred,u);r=real_step(real,u);errors.append(abs(p-r));pred=p;real=r
        predicted_return-=abs(1-p);actual_return-=abs(1-r)
    return {'predicted_return':predicted_return,'actual_return':actual_return,'return_gap':predicted_return-actual_return,'mean_transition_error':sum(errors)/len(errors)}
def exploitative(x,t):return 1.0
def conservative(x,t):return max(-.2,min(.2,1-x))
if __name__=='__main__':
    for name,p in [('aggressive',exploitative),('conservative',conservative)]:print(name,evaluate(p))

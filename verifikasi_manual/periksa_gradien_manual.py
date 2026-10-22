# Pemeriksa NUMERIK untuk lembar perhitungan manual.
# Bukan bukti bahwa pemilik repo menghitung semuanya sendiri.
# Jalankan: python verifikasi_manual/periksa_gradien_manual.py
import numpy as np
x = np.array([[1.0, 2.0]])
y = np.array([[1.0]])
W1 = np.array([[.1, .3], [.2, .4]])
b1 = np.array([[.1, .1]])
W2 = np.array([[.5], [-.4]])
b2 = np.array([[0.]])

def sigmoid(z):
    return 1/(1+np.exp(-z))

def forward_and_loss():
    z1=x@W1+b1
    a1=sigmoid(z1)
    z2=a1@W2+b2
    p=sigmoid(z2)
    L=np.mean(np.logaddexp(0,z2)-y*z2)
    return z1,a1,z2,p,float(L)

z1,a1,z2,p,L = forward_and_loss()
delta2=(p-y)/len(x)
grad_W2=a1.T@delta2
grad_b2=delta2.sum(axis=0, keepdims=True)
delta1=(delta2@W2.T)*a1*(1-a1)
grad_W1=x.T@delta1
grad_b1=delta1.sum(axis=0,keepdims=True)

for name,param,grad in [('W1',W1,grad_W1),('b1',b1,grad_b1),('W2',W2,grad_W2),('b2',b2,grad_b2)]:
    max_diff=0.0
    for idx in np.ndindex(param.shape):
        original=param[idx]
        eps=1e-5
        param[idx]=original+eps
        lp=forward_and_loss()[4]
        param[idx]=original-eps
        lm=forward_and_loss()[4]
        param[idx]=original
        num=(lp-lm)/(2*eps)
        max_diff=max(max_diff,abs(num-grad[idx]))
    print(f'{name}: beda maksimum = {max_diff:.12g}')
    assert max_diff<1e-8
print('Forward:', 'z1 =',z1,'a1 =',a1,'z2 =',z2,'p =',p,'loss =',L)
W1 -= grad_W1
b1 -= grad_b1
W2 -= grad_W2
b2 -= grad_b2
print('Sesudah pembaruan: p =',forward_and_loss()[3],'loss =',forward_and_loss()[4])
assert abs(forward_and_loss()[4]-0.29427123392738963)<1e-9
print('SEMUA PEMERIKSAAN NUMERIK LULUS')

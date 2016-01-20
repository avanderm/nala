function [ U, S, V ] = svd_power( A, k, tol )
%SVD_POWER Computes SVD components using a power-like method
%   In each iteration a singular value is computed and stored together with
%   left and right singular vectors. The data matrix A is then shrunk using
%   Householder reflectors to find the next singular components of the
%   spectrum of A. Subsequent left and right singular value must be
%   backtransformed using the inverse of the orthonormal Householder
%   reflectors.

[m,n] = size(A);

U = zeros(m,k);
V = zeros(n,k);
s = zeros(k,1);

sig = Inf;
for i=1:k
    v = mean(A)';
    v = v/norm(v);
    
    tau = Inf;
    while (tau > sqrt(m*n)*tol)
        u = A*v;
        sig2 = norm(u);
        tau = abs(sig - sig2)/sig2;
        sig = sig2;
        u = u/sig;
        
        v = (u'*A)';
        v = v/norm(v);
    end
    
    s(i) = sig;
    
    U(i:end,i) = u;
    V(i:end,i) = v;
    
    ur = u - sign(u(1))*eye(m-(i-1),1);
    ur = ur/norm(ur);
    
    vr = v - sign(v(1))*eye(n-(i-1),1);
    vr = vr/norm(vr);
    
    A = (eye(m-(i-1)) - 2*(ur*ur'))*A*(eye(n-(i-1)) - 2*(vr*vr'));
    A = A(2:end,2:end);
end

for i=k-1:-1:1
    bu = U(i:end,i) - sign(U(i,i))*eye(m-(i-1),1);
    bu = bu/norm(bu);
    
    U(i:end,i+1:end) = (eye(m-(i-1)) - 2*(bu*bu'))'*U(i:end,i+1:end);
    
    bv = V(i:end,i) - sign(V(i,i))*eye(n-(i-1),1);
    bv = bv/norm(bv);
    
    V(i:end,i+1:end) = (eye(n-(i-1)) - 2*(bv*bv'))'*V(i:end,i+1:end);
end

S = diag(s);

end
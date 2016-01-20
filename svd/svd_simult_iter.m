function [ s ] = svd_simult_iter( A, tol )
%SVD_SIMULT_ITER Computes the SVD of A
%   Idea of simultaneous iteration extended to computing the SVD of a matrix A

[m,n] = size(A);

if m < n
    A = A';
end

h = min(m,n);
Qr = eye(h);

tau = Inf;

while tau > sqrt(m*n)*tol
    L = A*Qr;
    [Ql,~] = qr(L,0);
    
    R = Ql'*A;
    [Qr,Rr] = qr(R',0);
    
    tau = norm(triu(Rr,1));
end

s = abs(diag(Rr));

end

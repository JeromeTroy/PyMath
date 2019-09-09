n = 512;
A = rand(n);
b = rand(n, 1);

for i = 1:length(A)
    A(i, 1:i-1) = 0;
end

x = A \ b;

error = abs(A * x - b);

semilogy(error, 'o');
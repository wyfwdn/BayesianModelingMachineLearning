clear;
close all;
load data.mat;
%setup the parameters and initialize some parameters with zeros matrix% 
K=10;d=2;n=250;
phi=zeros(n,K);n1=zeros(K);pi=zeros(K);sigma=zeros(d,d,K);
f=zeros(100,1);phix=zeros(d);sum_phi=zeros(K);
[lb,mu]=kmeans(X',K);mu1=mu';X1=X';
%initialize sigma and pi%
for j=1:K
    sigma(:,:,j)=cov(X1(lb==j,:));
    pi(j) = sum(lb==j)/n;
end
%begin iteration of EM%
for t=1:100 %100 iterations%
%E-step: set phi(i,j) at iteration t%
for j=1:K
for i=1:n
Normal=mvnpdf(X(:,i),mu1(:,j),sigma(:,:,j));
phi(i,j)=pi(j).*Normal;
end
end
%calculate ft to assess convergence of f as a function of t%
f(t)=sum(log(sum(phi,2)));
phi=phi./repmat(sum(phi,2),1,K);%normalization%
%M-step: update n,mu,sigma,pi%
for j=1:K
sum_phi=sum(phi,1);
n1(j)=sum_phi(j);
sum_phix=zeros();
for i=1:n
phix=phi(i,j).*X(:,i);
sum_phix=phix+sum_phix;
end
mu1(:,j)=sum_phix./n1(j);
sum_phixmu=zeros(d,d);
for i=1:n
sum_phixmu=sum_phixmu+phi(i,j).*((X(:,i)-mu1(:,j))*(X(:,i)-mu1(:,j))');
end 
sigma(:,:,j)=sum_phixmu./n1(j);
pi(j)=n1(j)/n;
end
end
%plot part(b) graph%
plot(f);
[maxphi,L] = max(phi');
color = ['b','r','g','c','m','b','r','g','c','m'];
dot = ['.','.','.','.','.','o','o','o','o','o'];
figure;
for i=1:n
    plot(X(1,i),X(2,i),[color(L(i)) dot(L(i))]) 
    hold on
end
plot(mu1(1,:),mu1(2,:),['k','x']);%plot part(c) graph%
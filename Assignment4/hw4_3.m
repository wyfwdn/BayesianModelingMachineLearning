clear;
close all
load data.mat
%initialize parameters and matrix%
d=2;n=250;alpha=1;K=1;T=5;
X1=X';lb=ones(n,1);
m0=mean(X1)';c0 =0.1;a0=d;
A0=cov(X(1,:),X(2,:));B0=c0*d.*A0;
for t=1:T
s=zeros(n,1);sum_x=zeros(d,1);m=zeros(d,1);
c=zeros(n,1);a=zeros(n,1);
B=zeros(d, d, n);sigma=zeros(d, d, n);
phi=zeros(n,1);phi1=zeros(n,1);
mu=zeros(d,1);
for j=1:K
n1=zeros(n,1);
%accumulate the size of the cluster%
for i=1:n
if(lb(i)==j)
s(j)=s(j)+1;
end
n1(j)=s(j)-1;
if (lb(i)==j)
sum_x=sum_x+X(:,i);
end
end
x_ba=sum_x./s(j);
m(:,j)=c0/(c0 + s(j)).*m0+1/(c0+s(j)).*sum_x;
c(j)=s(j) + c0;
a(j)=s(j) + a0;
sum_B=zeros(d,d,n);
for i=1:n
if (lb(i)==j)
sum_B(:,:,j)=sum_B(:,:,j)+(X(:,i)-x_ba)*(X(:,i)-x_ba)';
end
end
B(:,:,j)=B0+sum_B(:,:,j)+s(j)/(a(j)*s(j)+1).*(x_ba-m(:,j))*(x_ba-m(:,j))';
sigma(:,:,j)=wishrnd(inv(B(:,:,j)),a(j));
mu(:,j)=mvnrnd(m(:,j),inv(c(j).*sigma(:,:,j)));
end
%calculate phi and new phi(for new cluster)%
for i=1:n
for j=1:K
phi(j)=mvnpdf(X(:,i),mu(:,j),inv(sigma(:,:,j))).*(n1(j)/(alpha+n-1));
temp1=c0/(pi*(1+c0))^(d/2);
temp2=det(B0+(c0/(1+c0).*(X(:,i)-m0)*(X(:,i)-m0)'))^(-0.5*(a0+1));
temp3=det(B0)^(-0.5*a0);
temp4=exp(gammaln((a0+1)/2)+gammaln(a0/2)-gammaln(a0/2)-gammaln((a0-1)/2));
phi1(j)=alpha/(alpha+n-1)*(temp1*temp2/temp3*temp4);
phi2=phi(j)/(phi(j)+phi1(j));
phi3=phi1(j)/(phi(j)+phi1(j));
end
c1=discretesample([phi2, phi3], 1);
if (c1==2)
K = K+1;
lb(i)=K;
x_ba=X(:,i);
sum_x=X(:,i);
m(:,K)=c0/(c0+1).*m0+1/(c0+1).*sum_x;
c(K)=1+c0;
a(K)=1+a0; 
sum_B(:,:,K)=(X(:,i)-x_ba)*(X(:,i)-x_ba)';
B(:,:,K)=B0+sum_B(:,:,K)+1/(a(K)+1).*(x_ba-m(:,K))*(x_ba-m(:,K))';
sigma(:,:,K)=wishrnd((B(:,:,K)),a(K));
mu(:,K)=mvnrnd(m(:,K),inv(c(K).*sigma(:,:,K)));
end  
end
end

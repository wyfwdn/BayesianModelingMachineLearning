clear;
close all
load data.mat
%set up the parameters and do the initialization%
K=2;%replace K here with other value from 2,4,10,25,100%
T=100;d=2;n=250;
t1=zeros(K);t2=zeros(K);t3=zeros(K);t4=zeros(K);
alpha0=ones(K);alpha=ones(K,1);
c=10;a=d.*ones(K);a0=d.*ones(K);
A=cov(X(1,:),X(2,:));
B0=d/10.*A;B=zeros(d,d,K);
sigma=zeros(d,d,K);n1=zeros(K);
phi=zeros(n,K);phix=zeros(d);
[lb,mu]=kmeans(X', K);
mu1=mu';X1=X';
%Initialize sigma and B%
for j=1:K
B(:,:,j)=B0;
sigma(:,:,j)=cov(X1(lb==j,:));
end
L=zeros(T,1);%Initialize objective function L%
for t=1:T %100 Iterations%
for j=1:K
for i=1:n
sum_psi=0;
for k=1:d
sum_psi=sum_psi+psi((1-k+a(j))/2);
end
%break down phi into t1 to t4%
t1(j)=sum_psi-log(det(B(:,:,j)));
t2(j)=(X(:,i)-mu1(:,j))'*(a(j).*pinv(B(:,:,j)))*(X(:,i)-mu1(:,j));
t3(j)=trace(a(j).*pinv(B(:,:,j))*sigma(:,:,j));
t4(j)=psi(alpha(j))-psi(sum(alpha));
phi(i,j)=exp(0.5*t1(j)-0.5*t2(j)-0.5*t3(j)+t4(j));
end  
end
phi=phi./repmat(sum(phi,2),1,K);  %Normalization%
for j=1:K       
sum_phi=sum(phi,1);
n1(j)=sum_phi(j); %update nj%
alpha(j)=alpha0(j)+n1(j);%update alpha(j)%
sigma(:,:,j)=pinv(1/c.*eye(d)+n1(j)*a(j).*pinv(B(:,:,j)));%update sigma(j)%
sum_phix=zeros(d,1);
for i=1:n
phix=phi(i,j).* X(:,i);
sum_phix=phix+sum_phix;
end
mu1(:,j)=sigma(:,:,j)*(a(j)*pinv(B(:,:,j))*sum_phix);%update qmu(j)%
%update q(sigma) by setting a and B%
a(j)=a0(j)+n1(j);
sum_phixmu=zeros(d,d);
for i=1:n
sum_phixmu=sum_phixmu+phi(i,j).*((X(:,i)-mu1(:,j))*(X(:,i)-mu1(:,j))'+sigma(:,:,j));
end      
B(:,:,j)=B0+sum_phixmu;
end
%Calculate the variational objective function%
sum_gamma=1;
for j=1:K
sum_gamma=sum_gamma+gammaln(alpha(j));
end
Eln1=0;Eln2=0;Eln3=0;Eln4=0;Eln5=0;Eln6=0;Eln7=0;Eln8=0;
for j=1:K
for i=1:n   
Eln1=Eln1+phi(i,j)*(0.5*t1(j)-0.5*t2(j)-0.5*t3(j)+t4(j));
Eln2=Eln2+phi(i,j)*t4(j);
Eln5=Eln5+phi(i,j)*log(phi(i,j));
end
Eln3=Eln3-c/2*trace(sigma(:,:,j)+mu1(:,j)*mu1(:,j)');
Eln4=Eln4+(a0(j)-d-1)/2*t1(j)+0.5*trace(B0*a(j).*pinv(B(:,:,j)));
Eln6=Eln6+0.5*log(det(pinv(sigma(:,:,j))));
Eln7=Eln7-0.5*log(2)*a(j)*d-log(gamma(0.5*(a(j)+1)))-log(gamma(0.5*a(j)))+0.5*a(j)*log(det(B(:,:,j)))+0.5*(a(j)-d+1)*t1(j)-0.5*trace(B(:,:,j)*a(j)*pinv(B(:,:,j)));
Eln8=Eln8+(alpha(j)-1)*t4(j);
end
L(t)=Eln1+Eln2+Eln3+Eln4-Eln5-Eln6-Eln7-Eln8+log(exp(gammaln(sum(alpha))-sum_gamma));
end


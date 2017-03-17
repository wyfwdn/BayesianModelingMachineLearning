w=zeros(15,1);
for t=1:1:100
sum0=0,sum1=0,sum2=0,sum3=0;
for L=1:1:11791
    sum1=sum1+Xtrain(:,L)*Xtrain(:,L)';
end


for L=1:1:11791
    if ytrain(:,L)==1
    phip=normpdf(Xtrain(:,L)'*w/-1.5);
    phi=normcdf(Xtrain(:,L)'*w/-1.5);
    Eq(L)=Xtrain(:,L)'*w+1.5*phip/(1-phi);

    else 
    phip=normpdf(Xtrain(:,L)'*w/-1.5);
    phi=normcdf(Xtrain(:,L)'*w/-1.5);
    Eq(L)=Xtrain(:,L)'*w+(-1.5)*phip/phi;
    end
end
for L=1:1:11791
    sum0=sum0+Xtrain(:,L)*Eq(L);
end
w=(sum1/2.25+eye(15))^(-1)*(sum0/2.25);
for L=1:1:11791
    sum2=sum2+ytrain(:,L)*log(normcdf(Xtrain(:,L)'*w/1.5));
    sum3=sum3+(1-ytrain(:,L))*log(1-normcdf(Xtrain(:,L)'*w/1.5));  
end 
lnp=log(1/(2*pi))*7.5-0.5*w'*w+sum2+sum3
stem(t,lnp) %plot the lnp as a function of t%
hold on;
end 

%make prediction to Xtest%
for L=1:1:1991
    if normcdf(Xtest(:,L)'*w/1.5)<0.5
        predict(L)=0;
    else predict(L)=1;
    end
end

%listing total numbers in a 2×2 table%
[C,order]=confusionmat(ytest,predict)

%find misclassified digits%
for I=1:1:982;
    if predict(I) > 0.5
        mis1=Xtest1(I,:),mis2=I;
    end
end

for J=983:1:1991;
    if predict(J) <0.5
        mis3=Xtest1(J,:),mis4=J;
    end
end

%find the three most ambiguous predictions%
for L=1:1:1991
    diff(L)=(predict(L)-0.5);
end
diff;

%Reconstruct the images%
x2=Q*w;
x3=reshape(x2,[28,28]) ;
imshow(x3);
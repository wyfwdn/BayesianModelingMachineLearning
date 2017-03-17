a=1; b=1; c=1; e=1; f=1;Xtrain1=Xtrain.';Xtest1=Xtest.';
N0=5842; 
N1=11791-5842;
m0=sum(Xtrain1(1:5842,:))/(N0+1/a); 
n0=N0+1/a;
p0=0.5*N0+b; q0=c+(sum(Xtrain1(1:5842,:).^2))/2-(((sum(Xtrain1(1:5842,:))).^2)/(N0+1/a))/2;
m1=sum(Xtrain1(5843:11791,:))/(N1+1/a); n1=N1+1/a;
p1=0.5*N1+b; q1=c+(sum(Xtrain1(5843:11791,:).^2))*0.5-(((sum(Xtrain1(5843:11791,:))).^2)/(N1+1/a))*0.5;
E0=2*q0*(1+n0)/n0; E1=2*q1*(1+n1)/n1;
prediction=1:1:1991;
integration0=ones(1991,15);integration1=ones(1991,15);
prior1=(e+N1)/(N0+N1+e+f);
prior0=(e+N0)/(N0+N1+e+f);
for L=1:1:1991
    
integration0(L,:)=exp(gammaln(p0+0.5)-gammaln(p0))*(2*pi*q0*(1+n0)/n0).^(-0.5).*(1+(Xtest1(L,:)-m0).^2./E0).^(-0.5-p0);
integration1(L,:)=exp(gammaln(p1+0.5)-gammaln(p1))*(2*pi*q1*(1+n1)/n1).^(-0.5).*(1+(Xtest1(L,:)-m1).^2./E1).^(-0.5-p1);
end
likelihood0=prod(integration0,2); 
likelihood1=prod(integration1,2); 
prediction0=likelihood0*prior0;
prediction1=likelihood1*prior1;
for L=1:1:1991
if prediction0(L,1) < prediction1(L,1)
    prediction(L)=1;
else prediction(L)=0;
end
end

%listing total numbers in a 2×2 table%
[C,order]=confusionmat(ytest,prediction); 

%find misclassified digits%
for I=1:1:982;
    if prediction0(I,1) < prediction1(I,1)
        mis1=Xtest1(I,:),mis2=I;
    end
end

for J=983:1:1991;
    if prediction0(J,1) > prediction1(J,1)
        mis3=Xtest1(J,:),mis4=J;
    end
end

%find the three most ambiguous predictions%
for L=1:1:1991
    diff(L)=(prediction0(L,1)-prediction1(L,1))/prediction0(L,1);
end
diff;

%Reconstruct the images%
 x2=Q*Xtest(:,L);
 x3=reshape(x2,[28,28]) ;
 imshow(x3);
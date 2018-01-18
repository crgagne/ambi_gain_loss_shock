
%14th Jan, adding one extra model, model 1A without the hidden info...

clear all; close all;
%subIDs={'S01','S04','S05','S06','S07','S09','S10','S11','S13', 'S14', 'S15', 'S16', 'S17', 'S19', 'S20','S25', 'S26', 'S27', 'S28', 'S29', 'S32', 'S33', 'S34', 'S35','S36', 'S37', 'S38', 'S39', 'S40','S41','S43'}; %Removed subject 'S42' from all subsequent analyses...

addpath('/vols/Data/bishop/Ambi/Scripts/BehaviourAnalysisScripts/BehaviouralAnalysisScripts/KTAnalysisScripts/');
addpath('/vols/Data/bishop/Ambi/Scripts/BehaviourAnalysisScripts/BehaviouralAnalysisScripts/');
addpath('/vols/Data/bishop/Ambi/Scripts/BehaviourAnalysisScripts/DataWorkspaces/');
addpath('/vols/Data/bishop/Ambi/Scripts/BehaviourAnalysisScripts/');

%The following Data Matrices should contain everything we need...(including STAI scores, pain scores etc)
    %For fmri
load('/vols/Data/bishop/Ambi/Scripts/BehaviourAnalysisScripts/DataWorkspaces/DataMatrix_AllfMRI_FinalSubjects'); %Note that this will need to be updated when get final STAI scores etc from Chris

%%Remove subject 31 from the data:
data(31)=[]; subIDs(31)=[]; pShock(:,31)=[]; choices(:,31)=[];

%This was to work out why the betas were different...
%load('/vols/Data/bishop/Ambi/Scripts/BehaviourAnalysisScripts/DataWorkspaces/TempDataTesting');

for s=1:length(subIDs)  
    subIDs{s}
        
    amb=data{s}(:,6);
    m = data{s}(:, [4 5]); 
    p = data{s}(:, [1 3]); %Using all trials
    %On ambiguous trials, need to replace the probability for the ambiguous urn with the beta corrected value...
    p(amb~=1,2)=pShock(:,s);
  
    %Flipping so choice always for left hand urn (Unambig if an unambig trial)
    choices_thissub=choices(:,s); choices_thissub(choices_thissub==55555)=NaN;
    choices_left=abs(1-choices_thissub); %flip 0 and 1...
    
     %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
    %%FINDING AA1...
    hidden_info=1-sqrt(amb);
    hidden_info_levels=unique(hidden_info);
    
            for i=1:length(hidden_info_levels)                 
                choices_hiddeninfolevel=choices_left(hidden_info==hidden_info_levels(i),:); %Bet for that info level for that subject -- this is in terms of choose ambig -- need to flip 
                    
                    %Propotion Chosen at each ambiguity level...Proportion chose UNAMBIG (left urn...)
                propChosen(i,s) = sum(choices_hiddeninfolevel(isnan(choices_hiddeninfolevel)==0))./length(choices_hiddeninfolevel(isnan(choices_hiddeninfolevel)==0)); %This is actually the propotion chosen at each ambiguity level...excluding ones where computer chose...
            end
     
    lm_aa1{s}=fitglm(hidden_info_levels, propChosen(:,s), 'linear');
    aa1(s)=table2array(lm_aa1{s}.Coefficients(2,1));
   
    %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
    
    %Making extra parameters...
    mdiff=diff(m,1,2); %Right hand side - left hand side (ambig-unambig), except for unambiguous trials (ambig==1), then it is just right-left
    pdiff=diff(p,1,2); %Right hand side - left hand side (ambig-unambig), except for unambiguous trials (ambig==1), then it is just right-left
    pdiff_log=sign(pdiff).*log(abs(pdiff) + 1);  %log modulus as can't take log of a negative number. So takes log while preserving sign.
    
    p_log=sign(p).*log(abs(p)+1);
    
    ev_diff_withlog=m(:,2).*p_log(:,2)-m(:,1).*p_log(:,1);  
    ev_diff_nolog=m(:,2).*p(:,2)-m(:,1).*p(:,1);
    
    %%Response time...
    Real_decisiontimetotal_thissub=data{s}(:,9)+data{s}(:,12); %Time that the urns were shown before question mark appeared (12), plus how long they took to press once it did appear (9).
    Fromquestionmark_thissub=data{s}(:,9);
   
    Real_decisiontimetotal_thissub_a=Real_decisiontimetotal_thissub(amb~=1);
    Fromquestionmark_thissub_a=Fromquestionmark_thissub(amb~=1); 
    
    %ambig trials
    pdiff_a=pdiff(amb~=1);
    pdiff_log_a=pdiff_log(amb~=1);
    mdiff_a=mdiff(amb~=1);   
    hiddeninfo_a=hidden_info(amb~=1);
    choices_left_a=choices_left(amb~=1);
    amb_a=amb(amb~=1);    
    evdiff_withlog_a=ev_diff_withlog(amb~=1);
    evdiff_nolog_a=ev_diff_nolog(amb~=1);
        
    %unambig trials
    pdiff_u=pdiff(amb==1);
    pdiff_log_u=pdiff_log(amb==1);
    mdiff_u=mdiff(amb==1);
    choices_left_u=choices_left(amb==1);
    evdiff_withlog_u=ev_diff_withlog(amb==1);    
    evdiff_nolog_u=ev_diff_nolog(amb==1);
    
    %CHANGE HERE...
    brainer=(((p(:,2)-p(:,1)).*(m(:,2)-m(:,1)))<0); %If difference have signs in opposite directions, then it is brainer trial
    nobrainer=(((p(:,2)-p(:,1)).*(m(:,2)-m(:,1)))>0) + (((p(:,2)-p(:,1)).*(m(:,2)-m(:,1)))==0); %If difference have signs in same direction, then it is no brainer trial
    
    brainer_a=brainer(amb~=1);
    nobrainer_a=nobrainer(amb~=1);
    
    %want to split info, logmodpdiff, mdiff and choice into brainer and no brainer...
        %making so same size with zeros for other type so can do all in one regression...
    pdiff_log_a_brainer=pdiff_log_a(brainer_a==1); pdiff_log_a_nobrainer=pdiff_log_a(nobrainer_a==1);    
    mdiff_a_brainer=mdiff_a(brainer_a==1); mdiff_a_nobrainer=mdiff_a(nobrainer_a==1);   
    hiddeninfo_a_brainer=hiddeninfo_a(brainer_a==1); hiddeninfo_a_nobrainer=hiddeninfo_a(nobrainer_a==1);    
    choices_a_brainer=choices_left_a(brainer_a==1); choices_a_nobrainer=choices_left_a(nobrainer_a==1);
    
%     %p and m for unambiguous urn, ambiguous trials
%     pu=p(amb~=1,1); mu=m(amb~=1,1); %urn on left on ambiguous trials- have flipped ambig trials so ambig urn always on right...
%     %p and m for ambiguous urn, ambiguous trials
%     pa=p(amb~=1,2); ma=m(amb~=1,2);
%     
%     %Now we want pu and pa as the logmod version...
%     %%NOTE: this is different from logmod(pdiff)- can't just take the difference of these ps AFTER do the logmod and get the same thing...
%     pu=sign(pu).*log(abs(pu) + 1); pa=sign(pa).*log(abs(pa) + 1);

    %Normalise regressors (use emma function)
    %Crucially we don't have any ==0 placeholder trials here, or would have to do this differently... 
    [mdiff_a]=Emma_norm(mdiff_a,mdiff_a~=200); %%NOTE That just doing mdiff_a~=200 so will run my function, but will include all trials (as none actually =200)
    [pdiff_a]=Emma_norm(pdiff_a,pdiff_a~=200);
    [pdiff_log_a]=Emma_norm(pdiff_log_a,pdiff_log_a~=200); %Now have got rid of all of the other 0 ones (for unambig trials), so want to normalise with all trials...(except for split low, high)...
    [hiddeninfo_a]=Emma_norm(hiddeninfo_a, hiddeninfo_a~=200);
    
    [pdiff_log_u]=Emma_norm(pdiff_log_u,pdiff_log_u~=200);
    [mdiff_u]=Emma_norm(mdiff_u,mdiff_u~=200);
    
    [evdiff_withlog_a]=Emma_norm(evdiff_withlog_a,evdiff_withlog_a~=200);
    [evdiff_nolog_a]=Emma_norm(evdiff_nolog_a,evdiff_nolog_a~=200);
    [evdiff_withlog_u]=Emma_norm(evdiff_withlog_u,evdiff_withlog_u~=200);
    [evdiff_nolog_u]=Emma_norm(evdiff_nolog_u,evdiff_nolog_u~=200);
    
%     [pu]=Emma_norm(pu,pu~=200);
%     [mu]=Emma_norm(mu,mu~=200);
%     [pa]=Emma_norm(pa,pa~=200);
%     [ma]=Emma_norm(ma,ma~=200);   
    
    %normalising the brainer and no brainer ones   
    [mdiff_a_brainer]=Emma_norm(mdiff_a_brainer,mdiff_a_brainer~=200); 
    [mdiff_a_nobrainer]=Emma_norm(mdiff_a_nobrainer,mdiff_a_nobrainer~=200);
    
    [pdiff_log_a_brainer]=Emma_norm(pdiff_log_a_brainer,pdiff_log_a_brainer~=200); 
    [pdiff_log_a_nobrainer]=Emma_norm(pdiff_log_a_nobrainer,pdiff_log_a_nobrainer~=200);
    
    [hiddeninfo_a_brainer]=Emma_norm(hiddeninfo_a_brainer,hiddeninfo_a_brainer~=200); 
    [hiddeninfo_a_nobrainer]=Emma_norm(hiddeninfo_a_nobrainer,hiddeninfo_a_nobrainer~=200);
    
    %normalising reaction time measures:
    [Real_decisiontimetotal_thissub_a]=Emma_norm(Real_decisiontimetotal_thissub_a,Real_decisiontimetotal_thissub_a~=200);
    [Fromquestionmark_thissub_a]=Emma_norm(Fromquestionmark_thissub_a,Fromquestionmark_thissub_a~=200);
    
    %Now that individual terms are normalised we can find the interaction terms..
       %For model 3
    mdiffinfo_a=mdiff_a.*hiddeninfo_a; pdiffinfo_a=pdiff_log_a.*hiddeninfo_a;
    mdiffpdiff_a=mdiff_a.*pdiff_log_a; mdiffpdiff_u=mdiff_u.*pdiff_log_u;
    
    %Note that using the ~= 500 instead of ~=0 as all of these that ==0 are still included as only looking at trials that we actually want here...
    [mdiffinfo_a]=Emma_norm(mdiffinfo_a,mdiffinfo_a~=500);
    [pdiffinfo_a]=Emma_norm(pdiffinfo_a,pdiffinfo_a~=500);
    [mdiffpdiff_a]=Emma_norm(mdiffpdiff_a,mdiffpdiff_a~=500);
    [mdiffpdiff_u]=Emma_norm(mdiffpdiff_u,mdiffpdiff_u~=500);
        
   %%Adding 14th Jan:
   %Model 1A without hidden info (what they SHOULD be doing)
    table_model1a_NOhiddeninfo=table(mdiff_a,pdiff_log_a,choices_left_a);
    mdl_1a_NOhiddeninfo=fitglm(table_model1a_NOhiddeninfo,'Distribution', 'binomial', 'Link', 'logit');
    %keyboard;
    bic_1a_NOhiddeninfo(s)= mdl_1a_NOhiddeninfo.ModelCriterion.BIC;
    aic_1a_NOhiddeninfo(s)= mdl_1a_NOhiddeninfo.ModelCriterion.AIC;
    betas_1a_NOhiddeninfo(:,s)=table2array(mdl_1a_NOhiddeninfo.Coefficients(:,1));
    p_values_1a_NOhiddeninfo(:,s)=mdl_1a_NOhiddeninfo.Coefficients(:,4); %need to check if these are parametric... 
    
    %%Trying the fitglm model now...   
    %Model 1A, chose unambig, hidden info...
    table_model1a=table(mdiff_a,pdiff_log_a,hiddeninfo_a,choices_left_a);
    mdl_1a=fitglm(table_model1a,'Distribution', 'binomial', 'Link', 'logit');
    %keyboard;
    bic_1a(s)= mdl_1a.ModelCriterion.BIC;
    aic_1a(s)= mdl_1a.ModelCriterion.AIC;
    betas_1a(:,s)=table2array(mdl_1a.Coefficients(:,1));
    p_values_1a(:,s)=mdl_1a.Coefficients(:,4); %need to check if these are parametric...
    
    %Model 1A  %%%WITHOUT LOG
    table_model1a_nolog=table(mdiff_a,pdiff_a,hiddeninfo_a,choices_left_a);
    mdl_1a_nolog=fitglm(table_model1a_nolog,'Distribution', 'binomial', 'Link', 'logit');
    %keyboard;
    bic_1a_nolog(s)= mdl_1a_nolog.ModelCriterion.BIC;
    aic_1a_nolog(s)= mdl_1a_nolog.ModelCriterion.AIC;
    betas_1a_nolog(:,s)=table2array(mdl_1a_nolog.Coefficients(:,1));
    p_values_1a_nolog(:,s)=mdl_1a_nolog.Coefficients(:,4); %need to check if these are parametric...
    
    %Model 1U 
    table_model1u=table(mdiff_u,pdiff_log_u,choices_left_u);
    mdl_1u=fitglm(table_model1u,'Distribution', 'binomial', 'Link', 'logit');
    bic_1u(s)= mdl_1u.ModelCriterion.BIC;
    aic_1u(s)= mdl_1u.ModelCriterion.AIC;
    betas_1u(:,s)=table2array(mdl_1u.Coefficients(:,1));
    p_values_1u(:,s)=mdl_1u.Coefficients(:,4); %need to check if these are parametric...
    
    %Alternative model for 1A and 1U with EVdiff instead..    
    %%%%WITH logmod P
        %need to work out if can weight these subjectively at the same time as calculating the betas...
    table_model1aEV_withlog=table(evdiff_withlog_a,hiddeninfo_a,choices_left_a);
    mdl_1aEVwithlog=fitglm(table_model1aEV_withlog,'Distribution', 'binomial', 'Link', 'logit');
    bic_1aEVwithlog(s)= mdl_1aEVwithlog.ModelCriterion.BIC;
    aic_1aEVwithlog(s)= mdl_1aEVwithlog.ModelCriterion.AIC;
    betas_1aEVwithlog(:,s)=table2array(mdl_1aEVwithlog.Coefficients(:,1));
    p_values_1aEVwithlog(:,s)=mdl_1aEVwithlog.Coefficients(:,4); %need to check if these are parametric...
    
    %Model 1U with EVdiff...
    table_model1uEV_withlog=table(evdiff_withlog_u,choices_left_u);
    mdl_1uEVwithlog=fitglm(table_model1uEV_withlog,'Distribution', 'binomial', 'Link', 'logit');
    bic_1uEVwithlog(s)= mdl_1uEVwithlog.ModelCriterion.BIC;
    aic_1uEVwithlog(s)= mdl_1uEVwithlog.ModelCriterion.AIC;
    betas_1uEVwithlog(:,s)=table2array(mdl_1uEVwithlog.Coefficients(:,1));
    p_values_1uEVwithlog(:,s)=mdl_1uEVwithlog.Coefficients(:,4); %need to check if these are parametric...
    
    %%WITHOUT logmod P
    table_model1aEV_nolog=table(evdiff_nolog_a,hiddeninfo_a,choices_left_a);
    mdl_1aEVnolog=fitglm(table_model1aEV_nolog,'Distribution', 'binomial', 'Link', 'logit');
    bic_1aEVnolog(s)= mdl_1aEVnolog.ModelCriterion.BIC;
    aic_1aEVnolog(s)= mdl_1aEVnolog.ModelCriterion.AIC;
    betas_1aEVnolog(:,s)=table2array(mdl_1aEVnolog.Coefficients(:,1));
    p_values_1aEVnolog(:,s)=mdl_1aEVnolog.Coefficients(:,4); %need to check if these are parametric...
    
    %Model 1U with EVdiff...
    table_model1uEV_nolog=table(evdiff_nolog_u,choices_left_u);
    mdl_1uEVnolog=fitglm(table_model1uEV_nolog,'Distribution', 'binomial', 'Link', 'logit');
    bic_1uEVnolog(s)= mdl_1uEVnolog.ModelCriterion.BIC;
    aic_1uEVnolog(s)= mdl_1uEVnolog.ModelCriterion.AIC;
    betas_1uEVnolog(:,s)=table2array(mdl_1uEVnolog.Coefficients(:,1));
    p_values_1uEVnolog(:,s)=mdl_1uEVnolog.Coefficients(:,4); %need to check if these are parametric...   

    
    %%%%%%%%%%%%%%%%%%Models 2, where have Mdiff*Pdiff interaction only...%%%%%%%%%%%%%%%%%%%%
    %%Model 2A (with pdiff*mdiff interaction term only...
    table_model2a=table(mdiff_a,pdiff_log_a,hiddeninfo_a,mdiffpdiff_a, choices_left_a);
    mdl_2a=fitglm(table_model2a,'Distribution', 'binomial', 'Link', 'logit');
    bic_2a(s)= mdl_2a.ModelCriterion.BIC;
    aic_2a(s)= mdl_2a.ModelCriterion.AIC;
    betas_2a(:,s)=table2array(mdl_2a.Coefficients(:,1));
    p_values_2a(:,s)=mdl_2a.Coefficients(:,4);
    
    %%Model 2U (with pdiff*mdiff interaction term only...
    table_model2u=table(mdiff_u,pdiff_log_u, mdiffpdiff_u, choices_left_u);
    mdl_2u=fitglm(table_model2u,'Distribution', 'binomial', 'Link', 'logit');
    bic_2u(s)= mdl_2u.ModelCriterion.BIC;
    aic_2u(s)= mdl_2u.ModelCriterion.AIC;
    betas_2u(:,s)=table2array(mdl_2u.Coefficients(:,1));
    p_values_2u(:,s)=mdl_2u.Coefficients(:,4);   
    
    %%%%%%%%%%%%%%%%%%Models 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%Model 3A -- Mdiff and Pdiff with Info...    
    %Model 3A- using formula to specify interaction terms...
       %So for model 3A I would do
        %table(mdiff_a,pdiff_log_a,info_a,choices_a);
                %'choices_a ~ mdiff_a + pdiff_log_a + info_a + mdiff_a:info_a + pdiff_log_a:info_a;
    table_model3a=table(mdiff_a,pdiff_log_a,hiddeninfo_a,mdiffinfo_a,pdiffinfo_a, choices_left_a);
    %'choices_a ~ mdiff_a + pdiff_log_a + info_a + mdiff_a:info_a + pdiff_log_a:info_a', %used this when using a formula to define a model rather than just explicitly adding the interaction terms that I made...
    mdl_3a=fitglm(table_model3a,'Distribution', 'binomial', 'Link', 'logit');
    bic_3a(s)= mdl_3a.ModelCriterion.BIC;
    aic_3a(s)= mdl_3a.ModelCriterion.AIC;
    betas_3a(:,s)=table2array(mdl_3a.Coefficients(:,1));
    p_values_3a(:,s)=mdl_3a.Coefficients(:,4);
    
     
    %%%Extra analyses not using right now...
        %Model 1 with reaction time measures added- how does this change the
        %betas and the key correlations?
        table_model1a_rxntotal=table(mdiff_a,pdiff_log_a,hiddeninfo_a,Real_decisiontimetotal_thissub_a,choices_left_a);
        mdl_1a_rxntotal=fitglm(table_model1a_rxntotal,'Distribution', 'binomial', 'Link', 'logit');
        bic_1a_rxntotal(s)= mdl_1a_rxntotal.ModelCriterion.BIC;
        betas_1a_rxntotal(:,s)=table2array(mdl_1a_rxntotal.Coefficients(:,1));
        p_values_1a_rxntotal(:,s)=mdl_1a_rxntotal.Coefficients(:,4); %need to check if these are parametric...

        table_model1a_rxnquestion=table(mdiff_a,pdiff_log_a,hiddeninfo_a,Fromquestionmark_thissub_a,choices_left_a);
        mdl_1a_rxnquestion=fitglm(table_model1a_rxnquestion,'Distribution', 'binomial', 'Link', 'logit');
        bic_1a_rxnquestion(s)= mdl_1a_rxnquestion.ModelCriterion.BIC;
        betas_1a_rxnquestion(:,s)=table2array(mdl_1a_rxnquestion.Coefficients(:,1));
        p_values_1a_rxnquestion(:,s)=mdl_1a_rxnquestion.Coefficients(:,4); %need to check if these are parametric...

        %Comparison of betas for brainer and no brainer trials
        table_model1a_brainer=table(mdiff_a_brainer,pdiff_log_a_brainer,hiddeninfo_a_brainer,choices_a_brainer);
        mdl_1a_brainer=fitglm(table_model1a_brainer,'Distribution', 'binomial', 'Link', 'logit');
        bic_1a_brainer(s)= mdl_1a_brainer.ModelCriterion.BIC;
        betas_1a_brainer(:,s)=table2array(mdl_1a_brainer.Coefficients(:,1));
        p_values_1a_brainer(:,s)=mdl_1a_brainer.Coefficients(:,4); %need to check if these are parametric...

        table_model1a_nobrainer=table(mdiff_a_nobrainer,pdiff_log_a_nobrainer,hiddeninfo_a_nobrainer,choices_a_nobrainer);
        mdl_1a_nobrainer=fitglm(table_model1a_nobrainer,'Distribution', 'binomial', 'Link', 'logit');
        bic_1a_nobrainer(s)= mdl_1a_nobrainer.ModelCriterion.BIC;
        betas_1a_nobrainer(:,s)=table2array(mdl_1a_nobrainer.Coefficients(:,1));
        p_values_1a_nobrainer(:,s)=mdl_1a_nobrainer.Coefficients(:,4); %need to check if these are parametric...
end

keyboard;


%correlation of betas for model 1 without reaction time measures added, and one with reaction time measures added

[h_model1_model1rxntotal,p_model1_model1rxntotal]=corr(betas_1a',betas_1a_rxntotal(1:4,:)');

[h_model1_model1rxnquestion,p_model1_model1rxnquestion]=corr(betas_1a',betas_1a_rxnquestion(1:4,:)');

%correlation of betas from the two models with different measures of reaction time...
[h_model1rxntotal_model1rxnquestion,p_model1rxntotal_model1rxnquestion]=corr(betas_1a_rxntotal',betas_1a_rxnquestion');


%Comparing brainer and no brainer betas
for i=1:size(betas_1a_brainer,1)
    figure(i); plot(betas_1a_brainer(i,:),betas_1a_nobrainer(i,:),'ro');
end


%Need to look at whether betas significant at group level and whether they
%correlate with individual difference stuff...

%Parametric
for i=1:size(betas_1a,1)
    [H1a(i),P1a(i)]=ttest(betas_1a(i,:)); [P1a_WSRT(i)]=signrank(betas_1a(i,:));
end

for i=1:size(betas_1u,1)
    [H1u(i),P1u(i)]=ttest(betas_1u(i,:)); [P1u_WSRT(i)]=signrank(betas_1u(i,:));
end

for i=1:size(betas_3a,1)
    [H3a(i),P3a(i)]=ttest(betas_3a(i,:)); [P3a_WSRT(i)]=signrank(betas_3a(i,:));
end

for i=1:size(betas_1aEVwithlog,1)
    [H1aEV(i),P1aEV(i)]=ttest(betas_1aEVwithlog(i,:));  [P1aEV_WSRT(i)]=signrank(betas_1aEVwithlog(i,:));
end

for i=1:size(betas_1uEVwithlog,1)
    [H1uEV(i),P1uEV(i)]=ttest(betas_1uEVwithlog(i,:)); [P1uEV_WSRT(i)]=signrank(betas_1uEVwithlog(i,:));
end

%Correlations with individ difference measures
Factor_TB=[0.01097,-0.49452,1.11225,-0.18338,-0.05074,0.38996,-0.36495,-0.5496,0.90514,-0.75364,0.11452,-0.03443,0.68479,-0.43943,-0.03443,2.42111,0.68479,-0.30987,1.12549,1.7376,-0.19969,-0.89952,-1.08417,1.97734,-0.19662,-1.41469,0.85974,-1.30452,-0.9903,-1.24943, 0 , -1.46978]; %added 0 for 31 spot so could remove later...
individ_measures=[Trait_fmri' State_fmri' BDI_fmri' Factor_TSB' Factor_TB'];
individ_measures(31,:)=[];

    %Parametric
for i=1:size(betas_1a,1)
    X=[betas_1a(i,:)' individ_measures];
    [R,p]=corrcoef(X);
    r1a_indiv(:,i)=R(:,1);
    P1a_indiv(:,i)=p(:,1);  
end

for i=1:size(betas_1u,1)
    X=[betas_1u(i,:)' individ_measures];
    [R,p]=corrcoef(X);
    r1u_indiv(:,i)=R(:,1);
    P1u_indiv(:,i)=p(:,1);  
end

for i=1:size(betas_2a,1)
    X=[betas_2a(i,:)' individ_measures];
    [R,p]=corrcoef(X);
    r1a_indiv(:,i)=R(:,1);
    P1a_indiv(:,i)=p(:,1);  
end

for i=1:size(betas_2u,1)
    X=[betas_2u(i,:)' individ_measures];
    [R,p]=corrcoef(X);
    r1u_indiv(:,i)=R(:,1);
    P1u_indiv(:,i)=p(:,1);  
end

for i=1:size(betas_3a,1)
    X=[betas_3a(i,:)' individ_measures];
    [R,p]=corrcoef(X);
    r3a_indiv(:,i)=R(:,1);
    P3a_indiv(:,i)=p(:,1);  
end

for i=1:size(betas_1aEVwithlog,1)
    X=[betas_1aEVwithlog(i,:)' individ_measures];
    [R,p]=corrcoef(X);
    r1aEV_indiv(:,i)=R(:,1);
    P1aEV_indiv(:,i)=p(:,1);  
end

for i=1:size(betas_1uEVwithlog,1)
    X=[betas_1uEVwithlog(i,:)' individ_measures];
    [R,p]=corrcoef(X);
    r1uEV_indiv(:,i)=R(:,1);
    P1uEV_indiv(:,i)=p(:,1);  
end


%non parametric
for i=1:size(betas_1a,1)
    X=[betas_1a(i,:)' individ_measures];
    [R,p]=corr(X,'Type','Spearman');
    r1a_indiv_Spearman(:,i)=R(:,1);
    P1a_indiv_Spearman(:,i)=p(:,1);
end

for i=1:size(betas_1u,1)
    X=[betas_1u(i,:)' individ_measures];
    [R,p]=corr(X,'Type','Spearman');
    r1u_indiv_Spearman(:,i)=R(:,1);
    P1u_indiv_Spearman(:,i)=p(:,1);
end

for i=1:size(betas_2a,1)
    X=[betas_2a(i,:)' individ_measures];
    [R,p]=corr(X,'Type','Spearman');
    r1a_indiv_Spearman(:,i)=R(:,1);
    P1a_indiv_Spearman(:,i)=p(:,1);
end

for i=1:size(betas_2u,1)
    X=[betas_2u(i,:)' individ_measures];
    [R,p]=corr(X,'Type','Spearman');
    r1u_indiv_Spearman(:,i)=R(:,1);
    P1u_indiv_Spearman(:,i)=p(:,1);
end

for i=1:size(betas_3a,1)
    X=[betas_3a(i,:)' individ_measures];
    [R,p]=corr(X,'Type','Spearman');
    r3a_indiv_Spearman(:,i)=R(:,1);
    P3a_indiv_Spearman(:,i)=p(:,1);
end

for i=1:size(betas_1aEVwithlog,1)
    X=[betas_1aEVwithlog(i,:)' individ_measures];
    [R,p]=corr(X,'Type','Spearman');
    r1aEV_indiv_Spearman(:,i)=R(:,1);
    P1aEV_indiv_Spearman(:,i)=p(:,1);
end

for i=1:size(betas_1uEVwithlog,1)
    X=[betas_1uEVwithlog(i,:)' individ_measures];
    [R,p]=corr(X,'Type','Spearman');
    r1uEV_indiv_Spearman(:,i)=R(:,1);
    P1uEV_indiv_Spearman(:,i)=p(:,1);
end

save('/vols/Data/bishop/Ambi/Scripts/BehaviourAnalysisScripts/DataWorkspaces/FinalModels_Jan142018_ChoseLeft_HiddenInfo');
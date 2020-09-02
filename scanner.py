from scan_heavytails import scan_ibm, scan_glv_interactions, scan_glv_maxcap, \
scan_logistic, scan_diversity, scan_glv_immigration

N = 2

#file = 'scan_ibm.csv'
#scan_ibm(file)

file = 'test.csv'
#scan_glv_interactions(file, N=N)

#file = 'scan_glv_maxcap_constant.csv'
#scan_glv_maxcap(file, N=N)

#file = 'scan_logistic_constant.csv'
#scan_logistic(file, N=N)

#file = 'scan_glv_immigration_constant.csv'
scan_glv_immigration(file, N=N)

#file = 'scan_glv_diversity.csv'
#scan_diversity(file, N=N)
import pandas as pd
import numpy as np

def cluster_kmeans(df, k):
    m=df.shape[0] #number of training examples
    n=df.shape[1] #number of features. Here n=2
    total = 5
    lablestorage = []
    ssestorage = []
    for itr in range(0,total):
        
        #Step1 Assigned random cluster
        randcentersarray = np.random.choice(range(0,m), k, replace=False)
        currentcenterarray = randcentersarray[:]
        #print(randcentersarray)

        #Step2 making lable df and assigining random center lables
        labels = []
        val = 0
        for x in range(0,m):
            if x in randcentersarray:
                labels.append(val)
                val = val + 1
            else:
                labels.append('NaN')
        #print(labels)

        #Step3 calculate euclidiandistance of each point with lables

        mat = len(df) 
        dist_mat = np.zeros((mat,mat))
        dfnumpy = df.to_numpy()
        for x in range(mat):
            for y in range(x+1,mat):  
                dist_ling = np.linalg.norm(dfnumpy[x]-dfnumpy[y])
                dist_mat[x,y] = dist_mat[y,x] = dist_ling
        #print(dist_mat)

        #Step4 Choose the label which is clossest for all non lable points
        #select all rows from distance matrix for which lable assignment is to be done
        #assign it to the nearest lable
        rowscanner = []
        for x in range(0,m):
            if labels[x] == 'NaN':
                rowscanner.append(x)
            else:
                pass
        #print(rowscanner)

        nalabels = []
        SSE_temp = []
        for x in rowscanner:
            rowdistance = dist_mat[x,currentcenterarray]
            result = np.where(rowdistance == np.amin(rowdistance))
            nalabels.append(int(result[0]))
            mindis = np.amin(rowdistance)
            SSE_temp.append(mindis**2)
        #print(nalabels)
        SSE = np.sum(SSE_temp)
        #print(SSE)

        z=0
        for x in range(0,m):
            if labels[x] == 'NaN':
                labels[x] = nalabels[z]
                z = z+1
            else:
                pass
        #print(labels)
        dflabled = df.copy()
        #print(dflabled)
        SSE_loopfirstrun = SSE + 50
        SSE_loop = SSE
        #print(SSE_loopfirstrun)
        #print(SSE_loop)
        #count = 0
  
        while ((SSE_loopfirstrun - SSE_loop)/SSE_loopfirstrun) > 0.025:
        
            labels_df = pd.DataFrame(labels)
            #print(dflabled)
            dflabled['Labels'] = labels_df.iloc[:]

            newcentroidbase = dflabled.groupby(by = 'Labels').agg([np.mean] )
            newcentroidbase = newcentroidbase.xs('mean', axis=1, level =1 ,drop_level=True)
            #print(newcentroidbase)
            newcentroid = []
            rows = len(df)
            cols = k
            #print(cols)
            dist_mat_loop = np.zeros((rows,cols))
            #print(dist_mat_loop)
    #        newcentroidbasenp = newcentroidbase.to_numpy()
    #         for x in range(rows):
    #             for ind,val in np.ndenumerate(newcentroidbasenp): 
    #                 dist_lingloop = np.linalg.norm(dfnumpy[x]-val)
    #                 #print(ind[1])
    #                 #print(val)
    #                 #print(dist_lingloop)
    #                 dist_mat_loop[x,ind[1]] = dist_lingloop   
    #         print(dist_mat_loop)
            for x in range(rows):
                for ind,r in newcentroidbase.iterrows(): 
                    #print(ind)
                    obs_xloop = df.iloc[x,:]
                    #print(obs_xloop.shape)
                    obs_yloop = newcentroidbase.iloc[ind,:]
                    #print(obs_yloop.shape)
                    eucl_dist_loop = np.sqrt(np.sum((obs_xloop - obs_yloop)**2))
                    dist_mat_loop[x,ind] = eucl_dist_loop 
            #print(dist_mat_loop)

            newcentroids = []
            SSE_looptemp = []
            for x in range(0,m):
                rowdistanceloop = dist_mat_loop[x,]
                ##resultloop = int(np.where(rowdistanceloop == np.amin(rowdistanceloop))[0])
                ##resultloop = np.asarray(np.where(rowdistanceloop == np.amin(rowdistanceloop)))
                #print(rowdistanceloop[np.argmin(rowdistanceloop)])
                #print(rowdistanceloop)
                resultloop = np.nanargmin(rowdistanceloop)
                newcentroids.append(resultloop)
                mindisloop = rowdistanceloop[np.nanargmin(rowdistanceloop)]
                #print(mindisloop)
                SSE_looptemp.append(mindisloop**2)
            
            if len(set(newcentroids)) != k:
                break
            else:
                pass

            #print(newcentroids)    
            #count = count + 1
            SSE_loopfirstrun = SSE_loop  
            #print(SSE_loopfirstrun)
            SSE_loop = np.sum(SSE_looptemp)
            #print(SSE_loop)
            labels = newcentroids
            #print(labels)
                
                
        ssestorage.append(SSE_loop)
        lablestorage.append(labels)
    
    #print(lablestorage)
    #print(ssestorage)
    #(m,i) = min((v,i) for i,v in enumerate(ssestorage))
    #f = (m,i)[1]
    (itr,val) = min((i,v) for i,v in enumerate(ssestorage))
    #g = int((m,i)[0])
    #print(f)
    #print(lablestorage[f])
    
    finallables = pd.DataFrame(lablestorage[itr])
    finallables.index = df.index
    
    fianlsse = ssestorage[itr]
    return finallables,fianlsse
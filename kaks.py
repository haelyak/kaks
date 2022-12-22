import random,numpy,scipy.stats,math
from Org import *
from evoSim import *
from fitnessDict import *

def allSame(seqpos,popL):
    '''Is the position seqpos the same in all orgs in popL? Return
    boolean.'''
    firstseqbase=popL[0].alleleNumT[seqpos]
    for org in popL:
        if firstseqbase != org.alleleNumT[seqpos]:
            return False
    return True

def countSubsGen(popL,lastFixBaseL,aaSubCtL,synSubCtL):
    '''Count amino acid changing and synonymous substitutions in the
population from one generation and add the resulting values to
aaSubCtL and synSubCtL.'''
    dnaLen=popL[0].dnaLen
    for pos in range(dnaLen):
        firstOrgAllele = popL[0].alleleNumT[pos] # org 0 at this gen and pos
        if firstOrgAllele !=lastFixBaseL[pos] and allSame(pos,popL):
            # we have a substitution
            lastFixBaseL[pos]=firstOrgAllele
            if firstOrgAllele < 0:
                synSubCtL[pos]+=1
            else:
                aaSubCtL[pos]+=1
    return lastFixBaseL,aaSubCtL,synSubCtL

def countSubs(simL):
    '''Given a list containing all the generations of a simulation, count
amino acid changing and synonymous substitutions at each nucleotide position.'''
    dnaLen=simL[0][0].dnaLen
    aaSubCtL=[0]*dnaLen
    synSubCtL=[0]*dnaLen
    lastFixBaseL=[0]*dnaLen
    for gen in range(len(simL)):
        lastFixBaseL,aaSubCtL,synSubCtL=countSubsGen(simL[gen],lastFixBaseL,aaSubCtL,synSubCtL)
    return aaSubCtL,synSubCtL

def countSitesGen(popL,aaSiteCtL,synSiteCtL):
    '''Count amino acid and synonymous sites for each org in popL at each
codon position, and add it all into aaSiteCtL and synSiteCtL.'''
    for org in popL:
        tempAAL,tempSynL=org.countAASynSites()
        aaSiteCtL=elementWiseAdd(tempAAL,aaSiteCtL)
        synSiteCtL=elementWiseAdd(tempSynL,synSiteCtL)
    return aaSiteCtL,synSiteCtL

def countSites(simL):
    '''Count amino acid and synonymous sites for each codon
position in a simulation. Return as aaSiteCtL and synSiteCtL.'''
    aaSiteCtL=[0]*int(simL[0][0].dnaLen/3)
    synSiteCtL=[0]*int(simL[0][0].dnaLen/3)
    for popL in simL:
        aaSiteCtL,synSiteCtL=countSitesGen(popL,aaSiteCtL,synSiteCtL)
    # must divide site counts by numGens*popSize
    ngXps = 1.0 * len(simL)*len(simL[0])
    aaSiteCtL=[x/ngXps for x in aaSiteCtL]
    synSiteCtL=[x/ngXps for x in synSiteCtL]    
    return aaSiteCtL,synSiteCtL

def elementWiseAdd(sourceL,sinkL):
    '''Add the stuff in sourceL element-wise into sinkL'''
    for i in range(len(sinkL)):
        sinkL[i]+=sourceL[i]
    return sinkL

def elementWiseDiv(aL,bL):
    '''Divide elements in aL by elements in bL, return result.'''
    outL=[]
    for i in range(len(aL)):
        outL.append(1.0 * aL[i] / bL[i])
    return outL

def subsByCodon(nucL):
    '''Given a list of subs at nuc positions, sum them in groups of 3 to
get the number of nucleotide subs at each codon position.'''
    codonL=[]
    for i in range(0,len(nucL),3):
        codonL.append(sum(nucL[i:i+3]))
    return codonL

def reformat(simLL):
    '''Each of simLL's sublists are simulation output from one
simulation. Rearrange so sublists are all values at a single
nucleotide (or codon) position.'''
    newSubL=[]
    for i in range(len(simLL[0])): newSubL.append([])
    for L in simLL:
        for i in range(len(L)):
            newSubL[i].append(L[i])
    return newSubL

def kaksSim(startAllele,popSize,numGens,mutProb,numReps,fitnessD):
    '''Run evolutionary simulations which keep track of ka and ks.'''
    kaLL=[]
    ksLL=[]
    fgenAvFitL=[]
    lgenAvFitL=[]
    for rep in range(numReps):
        popL=startingPop(startAllele,popSize)
        simL=evoSimSelect(popL,popSize,numGens,mutProb,fitnessD)

        fgenAvFitL.append(avFit(simL[0],fitnessD))
        lgenAvFitL.append(avFit(simL[-1],fitnessD))

        aaSubCtL,synSubCtL=countSubs(simL)
        # divide sub counts by numGens to get the av num subs per
        # generation
        avAASubsL=[float(x)/numGens for x in aaSubCtL]
        avSynSubsL=[float(x)/numGens for x in synSubCtL]
        
        aaSiteCtL,synSiteCtL=countSites(simL)

        # get ka and ks
        kaL=elementWiseDiv(subsByCodon(avAASubsL),aaSiteCtL)
        ksL=elementWiseDiv(subsByCodon(avSynSubsL),synSiteCtL)
        kaLL.append(kaL)
        ksLL.append(ksL)
        
    return reformat(kaLL),reformat(ksLL),fgenAvFitL,lgenAvFitL

def avFit(popL,fitnessD):
    '''Print average fitness for a pop.'''
    sm=0
    for org in popL:
        sm+=org.fitness(fitnessD)
    return 1.0*sm/len(popL)

def confIntMean(L,confidence,formatString):
    '''Calculate confidence interval for the true mean of the population L
was sampled from, return as tuple of strings. Argument confidence
gives proportion of the distribution which should be contained within
the interval, e.g. 0.95 for a 95% interval.'''
    mn=numpy.mean(L)
    sd=numpy.std(L)
    if sd==0:
        return "NA","NA"
    else:
        sem=sd/math.sqrt(len(L))
        lw,hi=scipy.stats.norm.interval(confidence, loc=mn, scale=sem)
        lw=format(lw,formatString)
        hi=format(hi,formatString)
        return lw,hi

def formatEntry(L,formatString):
    '''Print mean and conf int for L.'''
    mn=numpy.mean(L)
    lw,hi=confIntMean(L,0.95,formatString)
    return format(mn,formatString)+" ("+lw+"-"+hi+")"

def printSummary(kaL,ksL,fgenAvFitL,lgenAvFitL,formatString):
    '''Print summary of substitution rates at each position.'''

    print("Ka and Ks at each codon position")
    for i in range(len(kaL)):
        print("  codon "+str(i))
        print('    Ka    '+formatEntry(kaL[i],formatString))
        print('    Ks    '+formatEntry(ksL[i],formatString))

    print('Average fitness')
    print('  First generation', format(1.0*sum(fgenAvFitL)/len(fgenAvFitL),".2f"))
    print('  Last  generation', format(1.0*sum(lgenAvFitL)/len(lgenAvFitL),".2f"))
    print()

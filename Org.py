import random
      
## support functions
def startingPop(dna,popSize):
    '''Create a population of identical organisms popSize large, each with
a genome sequence given by dna.'''
    startingAlleleNumT=(0,)*len(dna)
    popL=[]
    for i in range(popSize):
        popL.append(org(dna,startingAlleleNumT))
    return popL
        
def cullPop(popL, fitnessD):
    '''According to fitnessD, determine which organisms from popL survive
    to reproduce.'''
    survivorL=[]
    for indiv in popL:
        if random.random()<indiv.fitness(fitnessD):
            survivorL.append(indiv)
    return survivorL

def codonsOneStepAway(codon):
    '''Given codon (a string), returns all the codons one mutational
    step away.'''
    oneStepL=[]
    for pos in range(3):
        L = list(codon)
        for base in ['A','C','G','T']:
            if codon[pos] != base:
                L[pos] = base
                oneStepL.append("".join(L))
    return oneStepL


def createCodonSiteCountD(codonD):
    '''Given dictionary of codons, calculates the number of aa and syn
    sites in each.'''
    codonSiteCountD = {}
    for codon in codonD.keys():
        thisaa = codonD[codon]
        aaSites = 0.0
        synSites = 0.0
        for oneStepCodon in codonsOneStepAway(codon):
            if codonD[oneStepCodon]==thisaa:
                synSites += 1
            else:
                aaSites += 1
        aaSites /= 3.0 # each of the oneStep cases represents 1/3 of a site
        synSites /= 3.0
        codonSiteCountD[codon] = (aaSites,synSites)
    return codonSiteCountD


## org class
class org:
    """An organism class for evolution simulations."""

    # class variables, instantiated once on import
    mutCounter = 0 
    codonD = {'CTT': 'L', 'ATG': 'M', 'ACA': 'T', 'ACG': 'T', 'ATC': 'I', 'AAC': 'N', 'ATA': 'I', 'AGG': 'R', 'CCT': 'P', 'ACT': 'T', 'AGC': 'S', 'AAG': 'K', 'AGA': 'R', 'CAT': 'H', 'AAT': 'N', 'ATT': 'I', 'CTG': 'L', 'CTA': 'L', 'CTC': 'L', 'CAC': 'H', 'AAA': 'K', 'CCG': 'P', 'AGT': 'S', 'CCA': 'P', 'CAA': 'Q', 'CCC': 'P', 'TAT': 'Y', 'GGT': 'G', 'TGT': 'C', 'CGA': 'R', 'CAG': 'Q', 'TCT': 'S', 'GAT': 'D', 'CGG': 'R', 'TTT': 'F', 'TGC': 'C', 'GGG': 'G', 'TAG': '*', 'GGA': 'G', 'TAA': '*', 'GGC': 'G', 'TAC': 'Y', 'TTC': 'F', 'TCG': 'S', 'TTA': 'L', 'TTG': 'L', 'TCC': 'S', 'ACC': 'T', 'TCA': 'S', 'GCA': 'A', 'GTA': 'V', 'GCC': 'A', 'GTC': 'V', 'GCG': 'A', 'GTG': 'V', 'GAG': 'E', 'GTT': 'V', 'GCT': 'A', 'TGA': '*', 'GAC': 'D', 'CGT': 'R', 'TGG': 'W', 'GAA': 'E', 'CGC': 'R'}

    codonSiteCountD = createCodonSiteCountD(codonD) # give aa and syn sites per codon
  
    def __init__(self, dna,alleleNumT):
        
        if type(dna) != str:
            raise Exception("DNA must be a string.")
        if len(dna) % 3 != 0:
            raise Exception("DNA length must be a multiple of 3.")
        
        self.dna = dna
        self.dnaLen = len(self.dna)
        self.alleleNumT = alleleNumT # pos num means aa mut, neg means syn
        self.protein = self.translate()

    def __repr__(self):
        return("(dna:"+self.dna+",prot:"+self.protein+")")


    def translate(self):
        proteinL=[]
        for i in range(0,len(self.dna),3):
            proteinL.append(self.__class__.codonD[self.dna[i:i+3]])

        return "".join(proteinL)
            
    def mutateBase(self,base):
        if base == 'A':
            choices = 'GTC'
        elif base == 'C':
            choices = 'AGT'
        elif base == 'T':
            choices = 'AGC'
        else: # base == 'G':
            choices = 'ATC'
        return(random.choice(choices))
    
    def replicate(self, mutProb):
        '''Replicate organism with mutation.'''
        # convert mutProb from per base to per sequence rate.
        mutProb=mutProb*self.dnaLen
        #(We assume rate is low, and we ignore the prob of two
        # mutations arising at the same time)
        if random.random() < mutProb:
            mutPos=random.choice(range(self.dnaLen))
            newDNA=self.dna[:mutPos]+self.mutateBase(self.dna[mutPos])+self.dna[mutPos+1:]
            newAlleleNumT=self.getNewMutationNumber(mutPos,newDNA)
            return org(newDNA,newAlleleNumT)
        else:
            return org(self.dna,self.alleleNumT) # no mutation

    def fitness(self,fitnessD):
        '''Given a dictionary of fitnesses, calculate the fitness of this org.'''
        if self.protein not in fitnessD:
            return 0 # missing from fitnessD means fitness 0
        else:
            return fitnessD[self.protein]
        
    def getNewMutationNumber(self,mutPos,newDNA):
        '''We need to give this mutation a unique number and determine if its
synon or amino acid changing.'''
        self.__class__.mutCounter+=1 # increment class mutCounter variable
        thisMutNum=self.__class__.mutCounter
        codonSt=int(mutPos/3)*3 # get location of codon this mut occured in
        codonEnd=codonSt+3
        oldCodon=self.dna[codonSt:codonEnd]
        newCodon=newDNA[codonSt:codonEnd]
        if self.__class__.codonD[oldCodon]==self.__class__.codonD[newCodon]:
            # its a synonymous mutation, make number negative
            thisMutNum=-thisMutNum
        # it stays positive if amino acid changing mutation
        # create new numbering tuple
        newAlleleNumT = self.alleleNumT[:mutPos]+(thisMutNum,)+self.alleleNumT[mutPos+1:]
        return newAlleleNumT

    def countAASynSites(self):
        '''Count the number of synonymous and amino acid changing
sites. Return as lists.'''
        aaL=[]
        synL=[]
        for i in range(0,self.dnaLen,3):
            aaSites,synSites=self.__class__.codonSiteCountD[self.dna[i:i+3]]
            aaL.append(aaSites)
            synL.append(synSites)
        return aaL,synL

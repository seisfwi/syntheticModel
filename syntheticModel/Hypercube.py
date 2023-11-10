import copy
class axis:

    def __init__(self, **kw):
        """Axis
                defaults to n=1, o=0., d=1."""
        self.n = 1
        self.o = 0.
        self.d = 1.
        self.label = ""
        self.unit = ""
        if "n" in kw:
            self.n = kw["n"]
        if "o" in kw:
            self.o = kw["o"]
        if "d" in kw:
            self.d = kw["d"]
        if "label" in kw:
            self.label = kw["label"]
        if "unit" in kw:
            self.unit = kw["unit"]
        if "axis" in kw:
            self.n = kw["axis"].n
            self.o = kw["axis"].o
            self.d = kw["axis"].d
            self.label = kw["axis"].label
            self.unit = kw["axis"].unit

    def __repr__(self):
        """Define print method for class"""
        if self.unit!="":
            return "n=%d\to=%f\td=%f\tlabel=%s\tunit=%s"%(self.n,self.o,self.d,self.label,self.unit)
        elif self.label!="":
            return "n=%d\to=%f\td=%f\tlabel=%s"%(self.n,self.o,self.d,self.label)
        else:
            return "n=%d\to=%f\td=%f"%(self.n,self.o,self.d)


class hypercube:

    def __init__(self, **kw):
        """initialize with
          - axes=[] A list of Hypercube.axis
          - ns=[] An list of integers (optionally lists of os,ds,labels)
          - hypercube From another hypercube"""
        isSet = False
        self.axes = []
        if "axes" in kw:
            for ax in kw["axes"]:
                self.axes.append(axis(n=ax.n, o=ax.o, d=ax.d, label=ax.label))
                isSet = True
        elif "ns" in kw:
            for n in kw["ns"]:
                self.axes.append(axis(n=n))
                isSet = True
        if "os" in kw:
            for i in range(len(kw["os"]) - len(self.axes)):
                self.axes.append(axis(n=1))
            for i in range(len(kw["os"])):
                self.axes[i].o = kw["os"][i]
        if "ds" in kw:
            for i in range(len(kw["ds"]) - len(self.axes)):
                self.axes.append(axis(n=1))
            for i in range(len(kw["ds"])):
                self.axes[i].d = kw["ds"][i]
        if "labels" in kw:
            for i in range(len(kw["labels"]) - len(self.axes)):
                self.axes.append(axis(n=1))
            for i in range(len(kw["labels"])):
                self.axes[i].label = kw["labels"][i]
        if "units" in kw:
            for i in range(len(kw["units"]) - len(self.axes)):
                self.axes.append(axis(n=1))
            for i in range(len(kw["units"])):
                self.axes[i].unit = kw["units"][i]
        if "hypercube" in kw:
            for i in range(kw["hypercube"].getNdim()):
                a = axis(axis=kw["hypercube"].getAxis(i + 1))
                self.axes.append(a)

    def clone(self):
        """Clone hypercube"""
        return self.copy.deepCopy()

    
    def subCube(self,nw,fw,jw):
        """Return a sub-cube"""
        axes=[]
        for i in range(len(self.axes)):
            axes.append(axis(n=nw[i],o=self.axes[i].o+self.axes[i].d*fw[i],d=self.axes[i].d*jw[i],label=self.axes[i].label,unit=self.axes[i].unit))
        return hypercube(axes=axes)

    def getNdim(self):
        """Return the number of dimensions"""
        return len(self.axes)

    def getAxis(self, i):
        """Return an axis"""
        return self.axes[i - 1]

    def getN123(self):
        """Get the number of elements"""
        n123 = 1
        for ax in self.axes:
            n123 = n123 * ax.n
        return n123

    def getNs(self):
        """Get a list of the sizes of the axes"""
        ns = []
        for a in self.axes:
            ns.append(a.n)
        return ns

    def checkSame(self,hyper):
        """CHeck to see if hypercube is the same space"""
        same=True
        for iax in range(len(self.axes),len(hyper.axes)):
            if iax < len(self.axes) and iax < len(hyper.axes):
                if self.axes[i].n!= hyper.axes[i].n:
                    return False
                elif abs((self.axes[i].o-hyper.axes[i].o)/max(.001,abs(self.axes[i].o))) > .001:
                    return False
                elif abs((self.axes[i].d-hyper.axes[i].d)/max(.001,abs(self.axes[i].d))) >.001:
                    return False
            elif iax < len(self.axes):
                if self.axes[iax].n !=1:
                    return False
            elif hyper.axes[iax].n !=1:
                return False
            return True

    def __repr__(self):
        """Define print method for hypercube class"""
        x=""
        for i in range(len(self.axes)):
            x+="Axis %d: %s\n"%(i+1,str(self.axes[i]))
        return x

    def addAxis(self, axis):
        """Add an axis to the hypercube"""
        self.axes.append(axis)

    
lst_func = []

# ----- Eq011



def for_eq011f(x):
    return 2**x + x - 5

def getEq011_f()->tuple:
    nnn = 100
    name_ = 'function 2**x + x = 5'
    return (nnn, for_eq011f, name_)

# ----- Eq012
def for_eq012f(x):
    return (3+x)/2022 + (2+x)/2023 + (1+x)/2024 + x/2025 + 4

def for_eq013f(x):
    if (type(x)==int) or (type(x)==float):
        if x == 0:
            return 1e50
        else:
            return 1 / x + 1 / (1 + x) - 1
    else:
        if x.any == 0:
            return 1e500
        else:
            return 1/x + 1/(1+x) - 1


def for_eq014f_pr(x):
    return (77**x - 121**x)

def for_eq015f(x):
    return (x**(2/3) - 9*x**(1/3)+8)


def for_eq014f(x):
    top =  (7**x - 11**x)
    bot = (77**x - 121**x)
    return top / (bot ** 0.5) - 1
    # if (type(bot)==int) or (type(bot)==float):
    #     if bot == 0:
    #         return None
    #     else:
    #         return top/(bot**0.5) - 1
    # else:
    #     if bot.any ==0:
    #         if bot == 0:
    #             return None
    #         else:
    #             return top/(bot**0.5) - 1



def getEq012_f()->tuple:
    nnn = 3300
    min_ = - nnn
    max_ = nnn
    name_ = 'function (3+x)/2022 + (2+x)/2023 + (1+x)/2024 + x/2025 = -4'
    return (nnn, min_,max_, for_eq013f, name_)

def getEq013_f()->tuple:
    nnn = 3300
    min_ = - nnn
    max_ = nnn
    name_ = 'function 1/x + 1/(1+x) = 1'
    return (nnn, min_,max_, for_eq013f, name_)

def getEq014_f()->tuple:
    nnn = 3300
    min_ = -50
    max_ = 50
    name_ = 'function (7**x - 11**x)/(77**x - 121**x)**0.5 - 1'
    return (nnn, min_,max_, for_eq014f, name_)

def getEq014_pr_f()->tuple:
    nnn = 3300
    min_ = -50
    max_ = 50
    name_ = 'function 77**x - 121**x'
    return (nnn, min_,max_, for_eq014f_pr, name_)

def getEq015_f()->tuple:
    nnn = 6300
    min_ = -600
    max_ = 600
    name_ = 'x**(2/3) - 9*x**(1/3)+8'
    return (nnn, min_,max_, for_eq015f, name_)

def for_eq015f_test():
    m = 1;    print(m,' ',for_eq015f(m) )
    m = 512;  print(m,' ',for_eq015f(m) )


def create_function_list():
    # lst_func.append(getEq011_f())
    # lst_func.append(getEq012_f())
    # lst_func.append(getEq013_f())
    # lst_func.append(getEq014_pr_f())
    # lst_func.append(getEq014_f())
    lst_func.append(getEq015_f())
    print(lst_func)

if __name__ == "__main__":
    for_eq015f_test()
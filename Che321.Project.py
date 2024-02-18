import math
from math import e


def critical(Mc_s, M):
    if M == "T":

        Tc = [[] for i in range(5)]
        for i in range(5):
            for j in range(5): Tc[i].append((Mc_s[i] * Mc_s[j]) ** 0.5)
        return Tc

    if M == "V":
        Vc1 = []
        Vc2 = []
        Vc3 = []
        Vc4 = []
        Vc5 = []
        Vc = [Vc1, Vc2, Vc3, Vc4, Vc5]
        for i in range(5):
            for j in range(5): Vc[i].append(((Mc_s[i] ** (1 / 3) + Mc_s[j] ** (1 / 3)) / 2) ** 3)
        return Vc

    if M == "Z":
        Zc1 = []
        Zc2 = []
        Zc3 = []
        Zc4 = []
        Zc5 = []
        Zc = [Zc1, Zc2, Zc3, Zc4, Zc5]
        for i in range(5):
            for j in range(5): Zc[i].append((Mc_s[i] + Mc_s[j]) / 2)
        return Zc

    if M == "w":
        w1 = []
        w2 = []
        w3 = []
        w4 = []
        w5 = []
        w = [w1, w2, w3, w4, w5]
        for i in range(5):
            for j in range(5): w[i].append((Mc_s[i] + Mc_s[j]) / 2)
        return w


def P_critical(Zc, Vc, Tc):
    R = 83.14
    Pc1 = []
    Pc2 = []
    Pc3 = []
    Pc4 = []
    Pc5 = []
    Pc = [Pc1, Pc2, Pc3, Pc4, Pc5]
    for i in range(5):
        for j in range(5): Pc[i].append(Zc[i][j] * R * Tc[i][j] / Vc[i][j])

    return Pc


def Alpha_not(T, w, Tc, Pc):
    T += 273.15
    R = 83.14  # Based on the units of Tc and Pc
    B1 = []
    B2 = []
    B3 = []
    B4 = []
    B5 = []
    B = [B1, B2, B3, B4, B5]

    for i in range(5):
        for j in range(5):
            B_0 = 0.083 - (0.422 / (T / Tc[i][j]) ** 1.6)
            B_1 = 0.139 - (0.172 / (T / Tc[i][j]) ** 4.2)
            B[i].append((B_0 + (w[i][j] * B_1)) * R * Tc[i][j] / Pc[i][j])
    return B


def deldeldel(B):
    del1 = []
    del2 = []
    del3 = []
    del4 = []
    del5 = []
    delta = [del1, del2, del3, del4, del5]

    for i in range(5):
        for j in range(5): delta[i].append(2 * B[i][j] - B[i][i] - B[j][j])
    return delta


def a_aa__haaa(Vc_s, Zc_s, T, Tc_s):
    R = 1.9872
    a1 = [0, -161.88, 291.27, 2206.384, 1390.13]
    a2 = [583.11, 0, 107.38, 813.18, 1734.42]
    a3 = [1448.01, 469.55, 0, -124.785, 1675.934]
    a4 = [245.3636, -31.19, 1008.749, 0, 13.3126]
    a5 = [125.3825, 183.04, -228.442, 248.8769, 0]
    a = [a1, a2, a3, a4, a5]
    V = []
    for i in range(5): V.append(Vc_s[i] * (Zc_s[i] ** ((1 - (T + 273.15) / Tc_s[i]) ** (2 / 7))))
    A1 = []
    A2 = []
    A3 = []
    A4 = []
    A5 = []
    A = [A1, A2, A3, A4, A5]
    for i in range(5):
        for j in range(5): A[i].append(V[j] / V[i] * math.exp((-a[i][j]) / (R * (T + 273.15))))

    return A


def gamma_more_like_gamsız(num, x, A):
    gamma = []
    for i in range(num):
        sum1 = 0
        for j in range(num):
            sum1 += x[j] * A[i][j]

        sum2 = 0
        for k in range(num):
            sum3 = 0
            for j in range(num):
                sum3 += x[j] * A[k][j]
            sum2 += x[k] * A[k][i] / sum3

        gamma_i = math.exp(1 - math.log(sum1) - sum2)
        gamma.append(gamma_i)

    return gamma


def fiyuu(num, y, delta, B, P, P_sat, T):
    fi = []
    T += 273.15  # Convert to Kelvin if T is initially in Celsius
    R = 8314  # Based on the units of T, B and P , kPa.cm^3 / mol.K

    for i in range(num):
        sum1 = 0
        for j in range(num):
            for k in range(num):
                sum1 += y[j] * y[k] * (2 * delta[j][i] - delta[j][k])

        phi_i = math.exp(((B[i][i] * (P - P_sat[i])) + (0.5 * P * sum1)) / (R * T))
        fi.append(phi_i)

    return fi


def içime_içime_vursun_ritim_içime(yaz_dostum_s, yaz_dostum_s_n, num):
    subs = yaz_dostum_s
    pop = []

    for i in subs:

        if i == "acetone":
            pop.append(0)
        elif i == "methanol":
            pop.append(1)
        elif i == "water":
            pop.append(2)
        elif i == "methyl_acetate":
            pop.append(3)
        elif i == "benzene":
            pop.append(4)

    return pop


def popss(pop, yaz_dostum_s_n, num):
    eps = pop.copy()
    mol = []
    for i in range(num):
        s = eps.index(min(eps))
        mol.append(yaz_dostum_s_n[s])
        eps.remove(min(eps))
        yaz_dostum_s_n.pop(s)

    return mol


def kafa_zehir(M, pop):
    for j in range(4, -1, -1):
        for z in range(4, -1, -1):
            if z not in pop:
                M[j].pop(z)

    for i in range(4, -1, -1):
        if i not in pop:
            M.pop(i)

    return M


def pop_mini(M, pop):
    for i in range(4, -1, -1):
        if i not in pop:
            M.pop(i)
    return M


def norMALize(num, x):
    x_t = 0
    for i in range(num): x_t += x[i]
    for i in range(num): x[i] = x[i] / x_t
    return x


def sari_cizmeli_memet_aga():
    yaz_dostum_c = input("Calculation to be concluded (Bubble P/Bubble T/Dew P/Dew T): ")
    yaz_dostum_c_list = ["Bubble P", "Bubble T", "Dew P", "Dew T"]
    while yaz_dostum_c not in yaz_dostum_c_list:
        yaz_dostum_c = input("Calculation to be concluded !!!(Bubble P/Bubble T/Dew P/Dew T)!!!: ")
    yaz_dostum_f = input("The method of calculation (Raoult's Law/Modified Raoult's Law/Gamma-Phi Formulation): ")
    yaz_dostum_f_list = ["Raoult's Law", "Modified Raoult's Law", "Gamma-Phi Formulation"]
    while yaz_dostum_f not in yaz_dostum_f_list:
        yaz_dostum_f = input("The method of calculation !!!(Raoult's Law/Modified Raoult's Law/Gamma-Phi Formulation)!!!: ")
    yaz_dostum_v = "value"
    if yaz_dostum_c == "Bubble P" or yaz_dostum_c == "Dew P": yaz_dostum_v = float(input("Specified temperature value in C: "))
    if yaz_dostum_c == "Bubble T" or yaz_dostum_c == "Dew T": yaz_dostum_v = float(input("Specified pressure value in kPa: "))
    yaz_dostum_num = int(input("Species total number: "))
    while yaz_dostum_num > 5 or yaz_dostum_num < 1:
        yaz_dostum_num = int(input("Species total number !!!(in between 1-5)!!!: "))
    yaz_dostum_s = []
    yaz_dostum_s_n = []
    yaz_dostum_s_list = ["acetone", "methanol", "water", "methyl_acetate", "benzene"]
    flag = 0
    while flag == 0:
        for i in range(yaz_dostum_num):
            species = input("Specify your component(acetone/methanol/water/methyl_acetate/benzene): ")
            while species not in yaz_dostum_s_list:
                species = input("Specify your component!!!(acetone/methanol/water/methyl_acetate/benzene)!!!: ")
            if species not in yaz_dostum_s:
                yaz_dostum_s.append(species)
            else:
                species = input("YOU ALREADY WROTE THIS, WRITE ANOTHER ONE AGAIN!!!, AND IN A PROPER MANNER!!!")
                if species not in yaz_dostum_s: yaz_dostum_s.append(species)
            species_n = float(input("Enter mole fraction of the species (either liquid or vapour mole fraction, required one)(keep in mind that the sum should add up to 1): "))
            while True:
                if species_n <= 1:
                    break
                else:
                    species_n = input("ENTER A RELATABLE NUMBER!!!")
            yaz_dostum_s_n.append(species_n)
        total_n = 0
        for i in range(yaz_dostum_num): total_n += yaz_dostum_s_n[i]
        if total_n == 1: flag = 1
        else:
            print("MOLE FRACTIONS MUST ADD UP TO EXACTLY 1, REPEAT THE WHOLE PROCESS, AGAIN")

    num = int(yaz_dostum_num)
    pop = içime_içime_vursun_ritim_içime(yaz_dostum_s, yaz_dostum_s_n, num)

    A_sat = [14.3145, 16.5785, 16.3872, 14.2456, 13.7819]
    B_sat = [2756.22, 3638.27, 3885.70, 2662.78, 2726.81]
    C_sat = [228.060, 239.500, 230.170, 219.690, 217.572]
    A_sat = pop_mini(A_sat, pop)
    B_sat = pop_mini(B_sat, pop)
    C_sat = pop_mini(C_sat, pop)
    w_s = [0.307, 0.564, 0.345, 0.331, 0.210]
    Zc_s = [0.233, 0.224, 0.229, 0.257, 0.271]
    Vc_s = [209, 118, 55.9, 228, 259]
    Tc_s = [508.2, 512.6, 647.1, 506.6, 562.2]
    Zc_t = critical(Zc_s, "Z")
    Vc_t = critical(Vc_s, "V")
    Tc_t = critical(Tc_s, "T")
    Pc_t = P_critical(Zc_t, Vc_t, Tc_t)
    w_t = critical(w_s, "w")

    if yaz_dostum_c == "Bubble P":

        flag = 0
        T = yaz_dostum_v
        P = int
        y = []
        pop = içime_içime_vursun_ritim_içime(yaz_dostum_s, yaz_dostum_s_n, num)
        x = popss(pop, yaz_dostum_s_n, num)
        B_t = Alpha_not(T, w_t, Tc_t, Pc_t)
        delta_t = deldeldel(B_t)
        A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
        B = kafa_zehir(B_t, pop)
        delta = kafa_zehir(delta_t, pop)
        A = kafa_zehir(A_t, pop)

        P_sat = []
        for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T + C_sat[i])))

        if yaz_dostum_f == "Raoult's Law":
            P = 0
            for i in range(num): P += x[i] * P_sat[i]
            y = []
            for i in range(num): y.append(x[i] * P_sat[i] / P)

        if yaz_dostum_f == "Modified Raoult's Law":
            gamma = gamma_more_like_gamsız(num, x, A)
            P = 0
            for i in range(num): P += x[i] * gamma[i] * P_sat[i]
            while flag == 0:
                y = []
                P_i = P
                for i in range(num): y.append(x[i] * gamma[i] * P_sat[i] / P)
                P = 0
                for i in range(num): P += x[i] * gamma[i] * P_sat[i]
                if abs(P - P_i) <= 0.0001: flag = 1

        if yaz_dostum_f == "Gamma-Phi Formulation":
            flag = 0
            gamma = gamma_more_like_gamsız(num, x, A)

            P = 0
            fi = []
            for i in range(num): fi.append(1)
            for i in range(num): P += x[i] * gamma[i] * P_sat[i] / fi[i]


            while flag == 0:

                y = []
                P_i = P
                for i in range(num): y.append(x[i] * gamma[i] * P_sat[i] / (fi[i] * P_i))
                fi = fiyuu(num, y, delta, B, P_i, P_sat, T)
                intersum = 0
                for i in range(num): intersum += x[i] * gamma[i] * P_sat[i] / fi[i]
                P = intersum
                if abs(P - P_i) <= 0.0001:
                    flag = 1
                else:
                    flag = 0

        Bubble_P = P
        Bubble_y = y
        return Bubble_P, Bubble_y

    if yaz_dostum_c == "Bubble T":
        P = yaz_dostum_v
        T = int
        y = list
        pop = içime_içime_vursun_ritim_içime(yaz_dostum_s, yaz_dostum_s_n, num)
        x = popss(pop, yaz_dostum_s_n, num)
        flag = 0
        fi = []
        T_sat = []
        for i in range(num): fi.append(1)
        for i in range(num): T_sat.append(B_sat[i] / (A_sat[i] - math.log(P, e)) - C_sat[i])
        T = 0
        for i in range(num): T += x[i] * T_sat[i]

        if yaz_dostum_f == "Gamma-Phi Formulation":
            A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
            A = kafa_zehir(A_t, pop)
            P_sat = []
            for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T + C_sat[i])))
            gamma = gamma_more_like_gamsız(num, x, A)
            j = 0
            sum = 0
            for i in range(num): sum += x[i] * gamma[i] * P_sat[i] / (fi[i] * P_sat[j])
            P_j_sat = P / sum
            T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
            while flag == 0:
                B_t = Alpha_not(T, w_t, Tc_t, Pc_t)
                delta_t = deldeldel(B_t)
                A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
                B = kafa_zehir(B_t, pop)
                delta = kafa_zehir(delta_t, pop)
                A = kafa_zehir(A_t, pop)
                gamma = gamma_more_like_gamsız(num, x, A)
                y = []
                P_sat = []
                T_i = T
                for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T_i + C_sat[i])))
                for i in range(num): y.append(x[i] * gamma[i] * P_sat[i] / (fi[i] * P))
                fi = fiyuu(num, y, delta, B, P, P_sat, T_i)

                sum = 0
                for i in range(num): sum += x[i] * gamma[i] * P_sat[i] / (fi[i] * P_sat[j])
                P_j_sat = P / sum
                T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
                if abs(T - T_i) <= 0.0001: flag = 1

        if yaz_dostum_f == "Modified Raoult's Law":
            A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
            A = kafa_zehir(A_t, pop)
            P_sat = []
            for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T + C_sat[i])))
            gamma = gamma_more_like_gamsız(num, x, A)
            j = 0
            sum = 0
            for i in range(num): sum += x[i] * gamma[i] * P_sat[i] / (fi[i] * P_sat[j])
            P_j_sat = P / sum
            T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
            while flag == 0:
                A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
                A = kafa_zehir(A_t, pop)
                gamma = gamma_more_like_gamsız(num, x, A)
                y = []
                P_sat = []
                T_i = T
                for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T_i + C_sat[i])))
                for i in range(num): y.append(x[i] * gamma[i] * P_sat[i] / P)
                sum = 0
                for i in range(num): sum += x[i] * gamma[i] * P_sat[i] / P_sat[j]
                P_j_sat = P / sum
                T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
                if abs(T - T_i) <= 0.0001: flag = 1
        if yaz_dostum_f == "Raoult's Law":
            P_sat = []
            for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T + C_sat[i])))
            j = 0
            sum = 0
            for i in range(num): sum += x[i] * P_sat[i] / (fi[i] * P_sat[j])
            P_j_sat = P / sum
            T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
            y = []
            P_sat = []
            T_i = T
            for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T_i + C_sat[i])))
            for i in range(num): y.append(x[i] * P_sat[i] / P)

            sum = 0
            for i in range(num): sum += x[i] * P_sat[i] / P_sat[j]
            P_j_sat = P / sum
            T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]

        bub_T = T
        bub_y = y

        return bub_T, bub_y

    if yaz_dostum_c == "Dew P":
        T = yaz_dostum_v
        P = int
        x = []
        pop = içime_içime_vursun_ritim_içime(yaz_dostum_s, yaz_dostum_s_n, num)
        y = popss(pop, yaz_dostum_s_n, num)
        B_t = Alpha_not(T, w_t, Tc_t, Pc_t)
        delta_t = deldeldel(B_t)
        A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
        B = kafa_zehir(B_t, pop)
        delta = kafa_zehir(delta_t, pop)
        A = kafa_zehir(A_t, pop)
        A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
        A = kafa_zehir(A_t, pop)
        flag = 0
        flag2 = 0
        fi = []
        gamma = []
        P_sat = []
        for i in range(num): fi.append(1)
        for i in range(num): gamma.append(1)
        for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T + C_sat[i])))
        P_set = 0
        for i in range(num): P_set += y[i] * fi[i] / (gamma[i] * P_sat[i])
        P = 1 / P_set
        for i in range(num): x.append(y[i] * fi[i] * P / (gamma[i] * P_sat[i]))


        if yaz_dostum_f == "Gamma-Phi Formulation":

            gamma = gamma_more_like_gamsız(num, x, A)
            P_set = 0
            for i in range(num): P_set += y[i] * fi[i] / (gamma[i] * P_sat[i])
            P = 1 / P_set
            flag2 = 0

            while flag == 0:
                x = []
                P_i = P
                for i in range(num): x.append((y[i] * fi[i] * P_i) / (gamma[i] * P_sat[i]))
                fi = fiyuu(num, y, delta, B, P_i, P_sat, T)

                while flag2 == 0:
                    gamma_i = gamma
                    x = []
                    for i in range(num): x.append((y[i] * fi[i] * P_i) / (gamma[i] * P_sat[i]))
                    x = norMALize(num, x)
                    gamma = gamma_more_like_gamsız(num, x, A)
                    huf = 0
                    for i in range(num):
                        diff = (abs(gamma_i[i] - gamma[i]))
                        if diff > 0.0001:
                            huf += 1
                    if huf == 0: flag2 = 1
                flag2 = 0
                P_set = 0
                for i in range(num): P_set += y[i] * fi[i] / (gamma[i] * P_sat[i])
                P = 1 / P_set
                if abs(P - P_i) <= 0.0001: flag = 1
            flag = 0

        if yaz_dostum_f == "Modified Raoult's Law":
            gamma = gamma_more_like_gamsız(num, x, A)
            P_set = 0
            for i in range(num): P_set += y[i] / (gamma[i] * P_sat[i])
            P = 1 / P_set
            flag = 0
            x = []
            P_i = P
            while flag == 0:
                x = []
                P_i = P
                for i in range(num): x.append((y[i] * P_i) / (gamma[i] * P_sat[i]))
                while flag2 == 0:
                    gamma_i = gamma
                    x = []
                    for i in range(num): x.append((y[i] * P_i) / (gamma[i] * P_sat[i]))
                    x = norMALize(num, x)
                    gamma = gamma_more_like_gamsız(num, x, A)
                    huf = 0
                    for i in range(num):
                        diff = (abs(gamma_i[i] - gamma[i]))
                        if diff > 0.0001:
                            huf += 1
                    if huf == 0: flag2 = 1
                flag2 = 0
                P_set = 0
                for i in range(num): P_set += y[i] / (gamma[i] * P_sat[i])
                P = 1 / P_set
                if abs(P - P_i) <= 0.0001: flag = 1
            flag = 0

        if yaz_dostum_f == "Raoult's Law":
            P_set = 0
            for i in range(num): P_set += y[i] / P_sat[i]
            P = 1 / P_set
            flag = 0
            x = []
            P_i = P
            while flag == 0:
                x = []
                P_i = P
                for i in range(num): x.append((y[i] * P_i) / P_sat[i])

                x = []
                for i in range(num): x.append((y[i] * P_i) / P_sat[i])
                x = norMALize(num, x)

                P_set = 0
                for i in range(num): P_set += y[i] / P_sat[i]
                P = 1 / P_set
                if abs(P - P_i) <= 0.0001: flag = 1
            flag = 0

        dew_P = P
        dew_x = x

        return dew_P, dew_x

    if yaz_dostum_c == "Dew T":

        P = yaz_dostum_v
        T = int
        x = list
        pop = içime_içime_vursun_ritim_içime(yaz_dostum_s, yaz_dostum_s_n, num)
        y = popss(pop, yaz_dostum_s_n, num)
        flag = 0
        flag2 = 0
        fi = []
        gamma = []
        T_sat = []
        for i in range(num): fi.append(1)
        for i in range(num): gamma.append(1)
        for i in range(num): T_sat.append(B_sat[i] / (A_sat[i] - math.log(P, e)) - C_sat[i])
        T = 0
        for i in range(num): T += y[i] * T_sat[i]
        P_sat = []
        for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T + C_sat[i])))
        j = 0
        sum = 0
        for i in range(num): sum += y[i] * fi[i] * P_sat[j] / (gamma[i] * P_sat[i])
        P_j_sat = P * sum
        T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
        P_sat = []
        for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T + C_sat[i])))

        if yaz_dostum_f == "Gamma-Phi Formulation":
            B_t = Alpha_not(T, w_t, Tc_t, Pc_t)
            delta_t = deldeldel(B_t)
            A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
            B = kafa_zehir(B_t, pop)
            delta = kafa_zehir(delta_t, pop)
            A = kafa_zehir(A_t, pop)
            fiyuu(num, y, delta, B, P, P_sat, T)
            x = []
            for i in range(num): x.append((y[i] * fi[i] * P) / (gamma[i] * P_sat[i]))
            gamma = gamma_more_like_gamsız(num, x, A)
            sum = 0
            for i in range(num): sum += y[i] * fi[i] * P_sat[j] / (gamma[i] * P_sat[i])
            P_j_sat = P * sum
            T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]

            while flag == 0:
                B_t = Alpha_not(T, w_t, Tc_t, Pc_t)
                delta_t = deldeldel(B_t)
                A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
                B = kafa_zehir(B_t, pop)
                delta = kafa_zehir(delta_t, pop)
                A = kafa_zehir(A_t, pop)
                T_i = T
                P_sat = []
                for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T_i + C_sat[i])))
                fi = fiyuu(num, y, delta, B, P, P_sat, T_i)

                while flag2 == 0:
                    gamma_i = gamma
                    x = []
                    for i in range(num): x.append((y[i] * fi[i] * P) / (gamma[i] * P_sat[i]))
                    x = norMALize(num, x)
                    gamma = gamma_more_like_gamsız(num, x, A)
                    huf = 0
                    for i in range(num):
                        diff = (abs(gamma_i[i] - gamma[i]))
                        if diff > 0.0001:
                            huf += 1
                    if huf == 0: flag2 = 1
                flag2 = 0
                sum = 0
                for i in range(num): sum += y[i] * fi[i] * P_sat[j] / (gamma[i] * P_sat[i])
                P_j_sat = P * sum
                T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
                if abs(T - T_i) <= 0.0001: flag = 1
        if yaz_dostum_f == "Modified Raoult's Law":

            A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
            A = kafa_zehir(A_t, pop)
            x = []
            for i in range(num): x.append((y[i] * P) / (gamma[i] * P_sat[i]))
            gamma = gamma_more_like_gamsız(num, x, A)
            sum = 0
            for i in range(num): sum += y[i] * P_sat[j] / (gamma[i] * P_sat[i])
            P_j_sat = P * sum
            T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]

            while flag == 0:

                A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
                A = kafa_zehir(A_t, pop)
                T_i = T
                P_sat = []
                for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T_i + C_sat[i])))

                while flag2 == 0:
                    gamma_i = gamma
                    x = []
                    for i in range(num): x.append((y[i] * P) / (gamma[i] * P_sat[i]))
                    x = norMALize(num, x)
                    gamma = gamma_more_like_gamsız(num, x, A)
                    huf = 0
                    for i in range(num):
                        diff = (abs(gamma_i[i] - gamma[i]))
                        if diff > 0.0001:
                            huf += 1
                    if huf == 0: flag2 = 1
                flag2 = 0
                sum = 0
                for i in range(num): sum += y[i] * fi[i] * P_sat[j] / (gamma[i] * P_sat[i])
                P_j_sat = P * sum
                T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
                if abs(T - T_i) <= 0.0001: flag = 1
        if yaz_dostum_f == "Raoult's Law":

            x = []
            for i in range(num): x.append((y[i] * P) / P_sat[i])

            sum = 0
            for i in range(num): sum += y[i] * P_sat[j] / P_sat[i]
            P_j_sat = P * sum
            T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]

            while flag == 0:

                T_i = T
                P_sat = []
                for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T_i + C_sat[i])))

                x = []
                for i in range(num): x.append((y[i] * P) / P_sat[i])
                x = norMALize(num, x)

                sum = 0
                for i in range(num): sum += y[i] * P_sat[j] / P_sat[i]
                P_j_sat = P * sum
                T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
                if abs(T - T_i) <= 0.0001: flag = 1

        dew_T = T
        dew_x = x

        return dew_T, dew_x

    if yaz_dostum_c == "Flash Calculation":
        V = int
        x_n = list
        y_n = list

        if yaz_dostum_f == "Gamma-Phi Formulation":
            P = yaz_dostum_v
            T = int
            x = list
            pop = içime_içime_vursun_ritim_içime(yaz_dostum_s, yaz_dostum_s_n, num)
            z = popss(pop, yaz_dostum_s_n, num)
            y = list
            x = z
            flag = 0
            fi = []
            T_sat = []
            for i in range(num): fi.append(1)
            for i in range(num): T_sat.append(B_sat[i] / (A_sat[i] - math.log(P, e)) - C_sat[i])
            T = 0
            for i in range(num): T += x[i] * T_sat[i]
            A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
            A = kafa_zehir(A_t, pop)
            P_sat = []
            for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T + C_sat[i])))
            gamma = gamma_more_like_gamsız(num, x, A)
            j = 0
            sum = 0
            for i in range(num): sum += x[i] * gamma[i] * P_sat[i] / (fi[i] * P_sat[j])
            P_j_sat = P / sum
            T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
            while flag == 0:
                B_t = Alpha_not(T, w_t, Tc_t, Pc_t)
                delta_t = deldeldel(B_t)
                A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
                B = kafa_zehir(B_t, pop)
                delta = kafa_zehir(delta_t, pop)
                A = kafa_zehir(A_t, pop)
                gamma = gamma_more_like_gamsız(num, x, A)
                y = []
                P_sat = []
                T_i = T
                for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T_i + C_sat[i])))
                for i in range(num): y.append(x[i] * gamma[i] * P_sat[i] / (fi[i] * P))
                fi = fiyuu(num, y, delta, B, P, P_sat, T_i)

                sum = 0
                for i in range(num): sum += x[i] * gamma[i] * P_sat[i] / (fi[i] * P_sat[j])
                P_j_sat = P / sum
                T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
                if abs(T - T_i) <= 0.0001: flag = 1
            Bubble_T = T
            Bubble_gamma = gamma
            Bubble_fi = fi
            x = list
            y = z
            flag = 0
            flag2 = 0
            fi = []
            gamma = []
            T_sat = []
            for i in range(num): fi.append(1)
            for i in range(num): gamma.append(1)
            for i in range(num): T_sat.append(B_sat[i] / (A_sat[i] - math.log(P, e)) - C_sat[i])
            T = 0
            for i in range(num): T += y[i] * T_sat[i]
            P_sat = []
            for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T + C_sat[i])))
            j = 0
            sum = 0
            for i in range(num): sum += y[i] * fi[i] * P_sat[j] / (gamma[i] * P_sat[i])
            P_j_sat = P * sum
            T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
            P_sat = []
            for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T + C_sat[i])))

            B_t = Alpha_not(T, w_t, Tc_t, Pc_t)
            delta_t = deldeldel(B_t)
            A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
            B = kafa_zehir(B_t, pop)
            delta = kafa_zehir(delta_t, pop)
            A = kafa_zehir(A_t, pop)
            fiyuu(num, y, delta, B, P, P_sat, T)
            x = []
            for i in range(num): x.append((y[i] * fi[i] * P) / (gamma[i] * P_sat[i]))
            gamma = gamma_more_like_gamsız(num, x, A)
            sum = 0
            for i in range(num): sum += y[i] * fi[i] * P_sat[j] / (gamma[i] * P_sat[i])
            P_j_sat = P * sum
            T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]

            while flag == 0:
                B_t = Alpha_not(T, w_t, Tc_t, Pc_t)
                delta_t = deldeldel(B_t)
                A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
                B = kafa_zehir(B_t, pop)
                delta = kafa_zehir(delta_t, pop)
                A = kafa_zehir(A_t, pop)
                T_i = T
                P_sat = []
                for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T_i + C_sat[i])))
                fi = fiyuu(num, y, delta, B, P, P_sat, T_i)

                while flag2 == 0:
                    gamma_i = gamma
                    x = []
                    for i in range(num): x.append((y[i] * fi[i] * P) / (gamma[i] * P_sat[i]))
                    x = norMALize(num, x)
                    gamma = gamma_more_like_gamsız(num, x, A)
                    huf = 0
                    for i in range(num):
                        diff = (abs(gamma_i[i] - gamma[i]))
                        if diff > 0.0001:
                            huf += 1
                    if huf == 0: flag2 = 1
                flag2 = 0
                sum = 0
                for i in range(num): sum += y[i] * fi[i] * P_sat[j] / (gamma[i] * P_sat[i])
                P_j_sat = P * sum
                T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
                if abs(T - T_i) <= 0.0001: flag = 1
            Dew_T = T
            Dew_gamma = gamma
            Dew_fi = fi
            T = (Bubble_T + Dew_T) / 2
            P_sat = []
            for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T + C_sat[i])))
            gamma_g = []
            for i in range(num): gamma_g.append((Dew_gamma[i] + Bubble_gamma[i]) / 2)
            fi_g = []
            for i in range(num): fi_g.append((Dew_fi[i] + Bubble_fi[i]) / 2)

            V = 0.5
            flag = 0
            while flag == 0:
                V_i = V
                K = []


                for i in range(num): K.append(gamma_g[i] * P_sat[i] /(fi_g[i] * P))

                F = 0
                for i in range(num): F += z[i] * (K[i] - 1) / (1 + (V * (K[i] - 1)))
                F_dev = 0
                for i in range(num): F_dev -= z[i] * ((K[i] - 1) ** 2) / (1 + (V * (K[i] - 1))) ** 2
                V = V_i - F / F_dev
                x = []
                for i in range(num): x.append(z[i] / (1 + V * (K[i] - 1)))
                y = []
                for i in range(num): y.append(K[i] * x[i])

                B_t = Alpha_not(T, w_t, Tc_t, Pc_t)
                delta_t = deldeldel(B_t)
                A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
                B = kafa_zehir(B_t, pop)
                delta = kafa_zehir(delta_t, pop)
                A = kafa_zehir(A_t, pop)
                gamma_g = gamma_more_like_gamsız(num, x, A)
                fi_g = fiyuu(num, y, delta, B, P, P_sat, T)
                K = []
                for i in range(num): K.append(gamma_g[i] * P_sat[i] / (fi_g[i] * P))

                x_n = []
                for i in range(num): x_n.append(z[i] / (1 + V * (K[i] - 1)))
                y_n = []
                for i in range(num): y_n.append(K[i] * x[i])
                huf = 0

                for i in range(num):
                    if abs(x[i] - x_n[i]) > 0.0001:
                        huf += 1
                for i in range(num):
                    if abs(y[i] - y_n[i]) > 0.0001:
                        huf += 1
                if abs(V - V_i) > 0.0001:
                    huf += 1
                if huf == 0: flag += 1


        if yaz_dostum_f == "Modified Raoult's Law":
            P = yaz_dostum_v
            T = int
            x = list
            pop = içime_içime_vursun_ritim_içime(yaz_dostum_s, yaz_dostum_s_n, num)
            z = popss(pop, yaz_dostum_s_n, num)

            y = list
            pop = içime_içime_vursun_ritim_içime(yaz_dostum_s, yaz_dostum_s_n, num)
            x = z
            flag = 0
            fi = []
            T_sat = []
            for i in range(num): fi.append(1)
            for i in range(num): T_sat.append(B_sat[i] / (A_sat[i] - math.log(P, e)) - C_sat[i])
            T = 0
            for i in range(num): T += x[i] * T_sat[i]
            A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
            A = kafa_zehir(A_t, pop)
            P_sat = []
            for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T + C_sat[i])))
            gamma = gamma_more_like_gamsız(num, x, A)
            j = 0
            sum = 0
            for i in range(num): sum += x[i] * gamma[i] * P_sat[i] / (fi[i] * P_sat[j])
            P_j_sat = P / sum
            T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
            while flag == 0:
                A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
                A = kafa_zehir(A_t, pop)
                gamma = gamma_more_like_gamsız(num, x, A)
                y = []
                P_sat = []
                T_i = T
                for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T_i + C_sat[i])))
                for i in range(num): y.append(x[i] * gamma[i] * P_sat[i] / P)
                sum = 0
                for i in range(num): sum += x[i] * gamma[i] * P_sat[i] / P_sat[j]
                P_j_sat = P / sum
                T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
                if abs(T - T_i) <= 0.0001: flag = 1
            Bubble_T = T
            Bubble_gamma = gamma


            y = z
            flag = 0
            flag2 = 0
            fi = []
            gamma = []
            T_sat = []
            for i in range(num): fi.append(1)
            for i in range(num): gamma.append(1)
            for i in range(num): T_sat.append(B_sat[i] / (A_sat[i] - math.log(P, e)) - C_sat[i])
            T = 0
            for i in range(num): T += y[i] * T_sat[i]
            P_sat = []
            for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T + C_sat[i])))
            j = 0
            sum = 0
            for i in range(num): sum += y[i] * fi[i] * P_sat[j] / (gamma[i] * P_sat[i])
            P_j_sat = P * sum
            T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
            P_sat = []
            for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T + C_sat[i])))

            A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
            A = kafa_zehir(A_t, pop)
            x = []
            for i in range(num): x.append((y[i] * P) / (gamma[i] * P_sat[i]))
            gamma = gamma_more_like_gamsız(num, x, A)
            sum = 0
            for i in range(num): sum += y[i] * P_sat[j] / (gamma[i] * P_sat[i])
            P_j_sat = P * sum
            T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]

            while flag == 0:

                A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
                A = kafa_zehir(A_t, pop)
                T_i = T
                P_sat = []
                for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T_i + C_sat[i])))

                while flag2 == 0:
                    gamma_i = gamma
                    x = []
                    for i in range(num): x.append((y[i] * P) / (gamma[i] * P_sat[i]))
                    x = norMALize(num, x)
                    gamma = gamma_more_like_gamsız(num, x, A)
                    huf = 0
                    for i in range(num):
                        diff = (abs(gamma_i[i] - gamma[i]))
                        if diff > 0.0001:
                            huf += 1
                    if huf == 0: flag2 = 1
                flag2 = 0
                sum = 0
                for i in range(num): sum += y[i] * fi[i] * P_sat[j] / (gamma[i] * P_sat[i])
                P_j_sat = P * sum
                T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
                if abs(T - T_i) <= 0.0001: flag = 1
            Dew_T = T
            Dew_gamma = gamma

            T = (Bubble_T + Dew_T) / 2
            P_sat = []
            for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T + C_sat[i])))
            gamma_g = []
            for i in range(num): gamma_g.append((Dew_gamma[i] + Bubble_gamma[i]) / 2)

            V = 0.5
            flag = 0
            while flag == 0:
                V_i = V
                K = []

                for i in range(num): K.append(gamma_g[i] * P_sat[i] / P)

                F = 0
                for i in range(num): F += z[i] * (K[i] - 1) / (1 + (V * (K[i] - 1)))
                F_dev = 0
                for i in range(num): F_dev -= z[i] * ((K[i] - 1) ** 2) / (1 + (V * (K[i] - 1))) ** 2
                V = V - F / F_dev
                x = []
                for i in range(num): x.append(z[i] / (1 + V * (K[i] - 1)))
                y = []
                for i in range(num): y.append(K[i] * x[i])

                A_t = a_aa__haaa(Vc_s, Zc_s, T, Tc_s)
                A = kafa_zehir(A_t, pop)
                gamma_g = gamma_more_like_gamsız(num, x, A)

                K = []
                for i in range(num): K.append(gamma_g[i] * P_sat[i] / P)

                x_n = []
                for i in range(num): x_n.append(z[i] / (1 + V * (K[i] - 1)))
                y_n = []
                for i in range(num): y_n.append(K[i] * x[i])
                huf = 0

                for i in range(num):
                    if abs(x[i] - x_n[i]) > 0.0001:
                        huf += 1
                for i in range(num):
                    if abs(y[i] - y_n[i]) > 0.0001:
                        huf += 1
                if abs(V - V_i) > 0.0001:
                    huf += 1
                if huf == 0: flag += 1
        if yaz_dostum_f == "Raoult's Law":
            P = yaz_dostum_v
            T = int
            y = list
            pop = içime_içime_vursun_ritim_içime(yaz_dostum_s, yaz_dostum_s_n, num)
            z = popss(pop, yaz_dostum_s_n, num)
            x = z
            flag = 0
            fi = []
            T_sat = []
            for i in range(num): fi.append(1)
            for i in range(num): T_sat.append(B_sat[i] / (A_sat[i] - math.log(P, e)) - C_sat[i])
            T = 0
            for i in range(num): T += x[i] * T_sat[i]
            P = yaz_dostum_v
            T = int
            y = list
            x = z
            flag = 0
            fi = []
            T_sat = []
            for i in range(num): fi.append(1)
            for i in range(num): T_sat.append(B_sat[i] / (A_sat[i] - math.log(P, e)) - C_sat[i])
            T = 0
            for i in range(num): T += x[i] * T_sat[i]
            P_sat = []
            for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T + C_sat[i])))
            j = 0
            sum = 0
            for i in range(num): sum += x[i] * P_sat[i] / (fi[i] * P_sat[j])
            P_j_sat = P / sum
            T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
            y = []
            P_sat = []
            T_i = T
            for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T_i + C_sat[i])))
            for i in range(num): y.append(x[i] * P_sat[i] / P)

            sum = 0
            for i in range(num): sum += x[i] * P_sat[i] / P_sat[j]
            P_j_sat = P / sum
            T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
            Bubble_T = T


            x = list
            pop = içime_içime_vursun_ritim_içime(yaz_dostum_s, yaz_dostum_s_n, num)
            y = z
            flag = 0
            flag2 = 0
            fi = []
            gamma = []
            T_sat = []
            for i in range(num): fi.append(1)
            for i in range(num): gamma.append(1)
            for i in range(num): T_sat.append(B_sat[i] / (A_sat[i] - math.log(P, e)) - C_sat[i])
            T = 0
            for i in range(num): T += y[i] * T_sat[i]
            P_sat = []
            for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T + C_sat[i])))
            j = 0
            sum = 0
            for i in range(num): sum += y[i] * fi[i] * P_sat[j] / (gamma[i] * P_sat[i])
            P_j_sat = P * sum
            T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
            P_sat = []
            for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T + C_sat[i])))
            x = []
            for i in range(num): x.append((y[i] * P) / P_sat[i])

            sum = 0
            for i in range(num): sum += y[i] * P_sat[j] / P_sat[i]
            P_j_sat = P * sum
            T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]

            while flag == 0:

                T_i = T
                P_sat = []
                for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T_i + C_sat[i])))

                x = []
                for i in range(num): x.append((y[i] * P) / P_sat[i])
                x = norMALize(num, x)

                sum = 0
                for i in range(num): sum += y[i] * P_sat[j] / P_sat[i]
                P_j_sat = P * sum
                T = B_sat[j] / (A_sat[j] - math.log(P_j_sat, e)) - C_sat[j]
                if abs(T - T_i) <= 0.0001: flag = 1
            Dew_T = T
            T = (Dew_T + Bubble_T) / 2
            P_sat = []
            for i in range(num): P_sat.append(math.exp(A_sat[i] - B_sat[i] / (T + C_sat[i])))
            K = []
            for i in range(num): K.append(P_sat[i] / P)
            V = 0.5
            flag = 0
            while flag == 0:
                V_i = V

                F = 0
                for i in range(num): F += z[i] * (K[i] - 1) / (1 + (V * (K[i] - 1)))
                F_dev = 0
                for i in range(num): F_dev -= z[i] * ((K[i] - 1) ** 2) / (1 + (V * (K[i] - 1))) ** 2
                V = V - F / F_dev
                x_n = []
                for i in range(num): x_n.append(z[i] / (1 + V * (K[i] - 1)))
                y_n = []
                for i in range(num): y_n.append(K[i] * x[i])

                huf = 0
                if abs(V - V_i) > 0.0001:
                    huf += 1
                if huf == 0:
                    flag += 1

        x = norMALize(num, x_n)
        y = norMALize(num, y_n)
        V_f = V
        x_f = x_n
        y_f = y_n
        return V_f, x_f, y_f
print(sari_cizmeli_memet_aga())
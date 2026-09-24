# =============================================================================
# SIMULADOR FOTOVOLTAICO TÉCNICO-ECONÓMICO
# Copyright (C) 2026  Javier Medina Moreno
# Licencia: GNU AGPLv3
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 1. VARIABLES DE ENTRADA (A definir por el usuario)
# =============================================================================

# Parámetros del Proyecto
DURACION_TOTAL = 30                 # Años de vida útil del proyecto completo (ej. 25)

# Parámetros físicos y de rendimiento
N_PANELES_FIJO = 14                 # Número de Paneles
POTENCIA_PANEL_W = 540              # Potencia del panel en Wp
GARANTIA_PANELES_ANOS = 25          # Años de garantía de los paneles (ej. 25)

PERDIDA_EFICIENCIA_ANO_1 = 1/100    # Degradación del 1er año (ej. 0.02 para un 2%)
PERDIDA_EFICIENCIA_ANUAL = 0.35/100 # Degradación a partir del 2º año (ej. 0.005 para 0.5%/año)

N_BATERIAS = 0                      # Número de Baterías
CAPACIDAD_BATERIA_KWH = 5.76        # Tamaño de cada batería física en kWh

SOC_MIN_BATERIA_PCT = 10            # Porcentaje de carga mínimo de la batería (0-100)
PERDIDAS_BATERIA = 0.05             # Porcentaje de pérdida de energía de la batería "round-trip"

ALPHA_DEG_BATERIA = 0.025           # Coeficiente alpha (efecto del paso del tiempo t)
BETA_DEG_BATERIA = 6.5e-6           # Coeficiente beta (efecto de la energía acumulada E)

VIDA_UTIL_INVERSOR = 10             # Años de vida útil esperada del inversor (ej. 12)

# Costes iniciales
COSTE_FIJO_INST = 7493 #12073 #9783                # € (PRECIO TOTAL: instalación, paneles, inversor y batería)
COSTE_INVERSOR = 1200               # € Coste de compra e instalación del inversor
COSTE_BATERIAS = 0

# Costes de mantenimiento y Tasas
OPEX_PANEL_ANO = 0.0                # €/año por panel (Mantenimiento, limpieza)
COSTE_MENSUAL_BV = 0.0              # €/mes (Cuota de la batería virtual)
costes_fijos_mensuales = np.full(12, 16)  # € (Potencia, impuestos, alquiler de contador)

# Precio energía
precio_valle = 0.09
precio_llano = 0.12
precio_punta = 0.2
precio_venta = 0.04

año_base = 2025  # Año de referencia para el calendario de discriminación horaria

# Parámetros Económicos
DEDUCCION_IRPF = 3000               # Deducción en el IRPF en €
TASA_DESCUENTO = 0.03               # Tasa de descuento real anual

# Parámetros avanzados de la Batería
LAMBDA_KNEE_BATERIA = 15            # Factor de la caída de capacidad tras el umbral
GAMMA_KNEE_BATERIA = 0.05           # Factor de escala del colapso
UMBRAL_BASE_DEG_BATERIA = 0.28      # Umbral base de pérdida admisible (28%) antes del codo
SOH_MINIMO_OPERATIVO = 0.60         # 60% de retención de capacidad original (corte del BMS por subtensión)

# Importación de Datos de Consumo
FACTOR_INCREMENTO_CONSUMO = 2.5     #1.7
try:
    df_consumo = pd.read_csv('consumption.csv')
    df_consumo = df_consumo.iloc[:8760]
    consumo_base_horaria = pd.to_numeric(df_consumo['Consumo_kWh'], errors='coerce').fillna(0).values 
except FileNotFoundError:
    consumo_base_horaria = np.full(8760, 0.64)       
    print("Aviso: Archivo de Consumos no encontrado. Inicializando consumo_base_horaria a 0.64 kW.")

consumo_base_horaria = consumo_base_horaria * FACTOR_INCREMENTO_CONSUMO

# Importación de generación base desde PVGIS
try:
    df_pvgis = pd.read_csv('production.csv')
    df_pvgis = df_pvgis.iloc[:8760]
    gen_base_horaria = pd.to_numeric(df_pvgis['P'], errors='coerce').fillna(0).values / 1000.0
except FileNotFoundError:
    gen_base_horaria = np.zeros(8760)      
    print("Aviso: Archivo PVGIS no encontrado. Inicializando gen_base_horaria con ceros.")

CAPACIDAD_TOTAL_BATERIA_KWH = CAPACIDAD_BATERIA_KWH * N_BATERIAS

# =============================================================================
# 2. MÓDULOS DE COMPORTAMIENTO FÍSICO
# =============================================================================

def calcular_factor_degradacion(ano):
    if ano == 1:
        return 1.0 - PERDIDA_EFICIENCIA_ANO_1
    else:
        return 1.0 - PERDIDA_EFICIENCIA_ANO_1 - (PERDIDA_EFICIENCIA_ANUAL * (ano - 1))

def calcular_capacidad_bateria_efectiva(t_anos, E_acumulada_total, cap_inicial_total, soc_min_pct, n_baterias, alpha=ALPHA_DEG_BATERIA, beta=BETA_DEG_BATERIA):
    """
    Calcula la capacidad restante basándose en el tiempo, uso histórico acumulado
    y la dinámica a trozos (Piecewise) con colapso de resistencia interna.
    """
    if t_anos <= 0:
        return cap_inicial_total
        
    if n_baterias <= 0:
        return 0.0
        
    # Normalizamos la energía acumulada por el número de módulos físicos
    energia_acumulada_por_modulo = E_acumulada_total / n_baterias
        
    # Calcular el Umbral Dinámico penalizado por el límite de profundidad de descarga
    dod_maximo = (100 - soc_min_pct) / 100.0
    pen_dod = max(0, (dod_maximo - 0.80) * 0.05)
    umbral_dinamico = UMBRAL_BASE_DEG_BATERIA - pen_dod
    
    # Calcular la degradación base para el módulo (Fórmula: alpha*sqrt(t) + beta*E)
    degradacion = (alpha * np.sqrt(t_anos)) + (beta * energia_acumulada_por_modulo)
    
    # Lógica definida a trozos
    if degradacion <= umbral_dinamico:
        # Fase 1: Desgaste lineal y por calendario
        cap_efectiva_total = cap_inicial_total * (1 - degradacion)
    else:
        # Fase 2: Colapso por resistencia interna (Knee point)
        caida = cap_inicial_total * (1 - umbral_dinamico - GAMMA_KNEE_BATERIA * (np.exp(LAMBDA_KNEE_BATERIA * (degradacion - umbral_dinamico)) - 1))
        cap_efectiva_total = max(0.0, caida)
        
    # Fase 3: Muerte técnica (Corte de protección del BMS)
    if cap_efectiva_total < (cap_inicial_total * SOH_MINIMO_OPERATIVO):
        return 0.0
        
    return cap_efectiva_total

def calcular_balance_fisico(N, consumo, gen_unitaria_ajustada):
    kwp_instalados = (POTENCIA_PANEL_W / 1000.0) * N
    gen_total = gen_unitaria_ajustada * kwp_instalados
    
    autoconsumo = np.minimum(gen_total, consumo)
    excedentes = np.maximum(gen_total - consumo, 0)
    importado = np.maximum(consumo - gen_total, 0)
    
    return autoconsumo, excedentes, importado

# --- MÓDULO: Balance Físico con Batería ---
def calcular_balance_fisico_con_bateria(N, consumo, gen_unitaria_ajustada, cap_bateria_efectiva, soc_min_pct):
    kwp_instalados = (POTENCIA_PANEL_W / 1000.0) * N
    gen_total = gen_unitaria_ajustada * kwp_instalados
    
    autoconsumo_directo = np.minimum(gen_total, consumo)
    excedente_bruto = np.maximum(gen_total - consumo, 0)
    demanda_restante = np.maximum(consumo - gen_total, 0)
    
    uso_bateria = np.zeros(8760)
    excedentes_red = np.zeros(8760)
    importado_red = np.zeros(8760)
    soc_array = np.zeros(8760) 
    
    # Cálculo de la reserva física en kWh sobre la capacidad EFECTIVA de este año
    cap_minima_kwh = cap_bateria_efectiva * (soc_min_pct / 100.0)
    
    soc = cap_minima_kwh 
    perdida_parcial = PERDIDAS_BATERIA / 2.0
    
    for i in range(8760):
        # 1. Cargar la batería
        espacio_disponible = max(0, cap_bateria_efectiva - soc)
        max_carga_excedente = espacio_disponible / (1.0 - perdida_parcial) if (1.0 - perdida_parcial) > 0 else 0
        carga_desde_excedente = min(excedente_bruto[i], max_carga_excedente)
        energia_almacenada = carga_desde_excedente * (1.0 - perdida_parcial)
        soc += energia_almacenada
        excedentes_red[i] = excedente_bruto[i] - carga_desde_excedente
        
        # 2. Descargar la batería protegiendo la reserva mínima
        energia_disponible = max(0, soc - cap_minima_kwh)
        max_descarga_necesaria = demanda_restante[i] / (1.0 - perdida_parcial) if (1.0 - perdida_parcial) > 0 else 0
        energia_extraida = min(max_descarga_necesaria, energia_disponible)
        uso_bateria[i] = energia_extraida * (1.0 - perdida_parcial)
        soc -= energia_extraida
        importado_red[i] = demanda_restante[i] - uso_bateria[i]
        
        soc_array[i] = soc 
        
    autoconsumo_total = autoconsumo_directo + uso_bateria
    
    return autoconsumo_total, excedentes_red, importado_red, gen_total, soc_array, uso_bateria

def generar_array_tres_zonas(precio_valle, precio_llano, precio_punta, año_base):
    fechas = pd.date_range(start=f'{año_base}-01-01 00:00:00', 
                            end=f'{año_base}-12-31 23:00:00', 
                            freq='h')
    precios = np.full(8760, precio_valle)
    horas = fechas.hour
    dias_semana = fechas.weekday
    es_laborable = dias_semana < 5

    mask_llano = es_laborable & ((horas >= 8) & (horas < 10) | 
                                    (horas >= 14) & (horas < 18) | 
                                    (horas >= 22))
    precios[mask_llano] = precio_llano

    mask_punta = es_laborable & ((horas >= 10) & (horas < 14) | 
                                    (horas >= 18) & (horas < 22))
    precios[mask_punta] = precio_punta

    return precios

# =============================================================================
# 3. MÓDULOS ECONÓMICOS
# =============================================================================

def simular_facturacion_anual(autoconsumo, excedentes, importado, saldo_bolsa_inicial):
    dias_por_mes = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    horas_fin_mes = np.cumsum(dias_por_mes) * 24
    horas_inicio_mes = np.roll(horas_fin_mes, 1)
    horas_inicio_mes[0] = 0
    
    flujo_caja_anual = 0
    saldo_bolsa = saldo_bolsa_inicial
    
    for m in range(12):
        inicio = horas_inicio_mes[m]
        fin = horas_fin_mes[m]
        
        valor_excedentes = np.sum(excedentes[inicio:fin] * precio_venta_horario[inicio:fin])
        coste_importado = np.sum(importado[inicio:fin] * precio_compra_horario[inicio:fin])
        factura_pre_compensacion = coste_importado + costes_fijos_mensuales[m]
        
        saldo_disponible = saldo_bolsa + valor_excedentes
        descuento_aplicado = min(saldo_disponible, factura_pre_compensacion)
        saldo_bolsa = saldo_disponible - descuento_aplicado
        
        ahorro_directo = np.sum(autoconsumo[inicio:fin] * precio_compra_horario[inicio:fin])
        flujo_mensual = ahorro_directo + descuento_aplicado - COSTE_MENSUAL_BV
        
        flujo_caja_anual += flujo_mensual
        
    return flujo_caja_anual, saldo_bolsa

def calcular_van(N, consumo, gen_base):
    if N == 0: return 0.0

    capex_inicial = COSTE_FIJO_INST 
    van = -capex_inicial
    saldo_bolsa = 0.0  
    
    for t in range(1, DURACION_TOTAL + 1):
        factor_deg = calcular_factor_degradacion(t)
        gen_ano_t = gen_base * factor_deg
        
        auto, exc, imp = calcular_balance_fisico(N, consumo, gen_ano_t)
        flujo_anual, saldo_bolsa = simular_facturacion_anual(auto, exc, imp, saldo_bolsa)
        flujo_anual -= (OPEX_PANEL_ANO * N)
        
        if t == 1:
            flujo_anual += DEDUCCION_IRPF
        
        if t % VIDA_UTIL_INVERSOR == 0 and t < DURACION_TOTAL:
             flujo_anual -= COSTE_INVERSOR
        
        van += flujo_anual / ((1 + TASA_DESCUENTO) ** t)
        
    return van

# --- MÓDULO: Simulación Económica Completa ---
def calcular_simulacion_fija(N, cap_bateria_inicial, soc_min_pct, consumo, gen_base):
    if N == 0:
        return 0.0, 0.0, 0.0, np.zeros(DURACION_TOTAL + 1), np.zeros(8760), np.zeros(DURACION_TOTAL)
        
    capex_inicial = COSTE_FIJO_INST 
    
    van = -capex_inicial
    flujos_acumulados = [-capex_inicial]
    saldo_bolsa = 0.0  
    
    gen_total_ano1 = 0
    autoconsumo_total_ano1 = 0
    consumo_total_ano1 = 0
    soc_array_ano1 = np.zeros(8760)
    
    energia_acumulada_bateria = 0.0
    capacidades_bateria = []
    
    for t in range(1, DURACION_TOTAL + 1):
        factor_deg = calcular_factor_degradacion(t)
        gen_ano_t = gen_base * factor_deg
        
        # 1. Se calcula la capacidad de la batería para el año T en base al estrés acumulado hasta T-1
        cap_bateria_efectiva = calcular_capacidad_bateria_efectiva(
            t_anos=t, 
            E_acumulada_total=energia_acumulada_bateria, 
            cap_inicial_total=cap_bateria_inicial, 
            soc_min_pct=soc_min_pct,
            n_baterias=N_BATERIAS
        )
        capacidades_bateria.append(cap_bateria_efectiva)
        
        # 2. Se simula el balance horario del año entero con esta nueva capacidad "techo"
        auto, exc, imp, gen_total, soc_array, uso_bat = calcular_balance_fisico_con_bateria(N, consumo, gen_ano_t, cap_bateria_efectiva, soc_min_pct)
        
        # 3. Sumamos la energía procesada (throughput de descarga) a la historia para afectar al año siguiente
        # Solo sumamos uso si la batería sigue viva (aunque el array uso_bat ya vendría vacío si cap = 0)
        energia_acumulada_bateria += np.sum(uso_bat)
        
        if t == 1:
            gen_total_ano1 = np.sum(gen_total)
            autoconsumo_total_ano1 = np.sum(auto)
            consumo_total_ano1 = np.sum(consumo)
            soc_array_ano1 = soc_array 
            
        flujo_anual, saldo_bolsa = simular_facturacion_anual(auto, exc, imp, saldo_bolsa)
        flujo_anual -= (OPEX_PANEL_ANO * N)
        
        if t == 1:
            flujo_anual += DEDUCCION_IRPF
        
        if t % VIDA_UTIL_INVERSOR == 0 and t < DURACION_TOTAL:
             flujo_anual -= COSTE_INVERSOR
             
        van += flujo_anual / ((1 + TASA_DESCUENTO) ** t)
        flujos_acumulados.append(van)
        
    pct_autoconsumo = (autoconsumo_total_ano1 / gen_total_ano1) * 100 if gen_total_ano1 > 0 else 0.0
    pct_autosuficiencia = (autoconsumo_total_ano1 / consumo_total_ano1) * 100 if consumo_total_ano1 > 0 else 0.0
        
    return van, pct_autoconsumo, pct_autosuficiencia, flujos_acumulados, soc_array_ano1, np.array(capacidades_bateria)

# =============================================================================
# 4. OPTIMIZADOR PRINCIPAL Y DATOS BASE
# =============================================================================

def optimizar_sistema():
    array_paneles = np.arange(0, 14 + 1)
    resultados_van = []
    
    for N in array_paneles:
        van = calcular_van(N, consumo_base_horaria, gen_base_horaria)
        resultados_van.append(van)
        
    resultados_van = np.array(resultados_van)
    idx_optimo = np.argmax(resultados_van)
    n_optimo = array_paneles[idx_optimo]
    van_optimo = resultados_van[idx_optimo]
    
    print(f"--- RESULTADOS DE LA OPTIMIZACIÓN ---")
    print(f"Límite físico: 14 paneles.")
    print(f"Número óptimo recomendado: {n_optimo} paneles.")
    print(f"VAN proyectado a {DURACION_TOTAL} años: {van_optimo:.2f} €")
    
    plt.figure(figsize=(10, 6))
    plt.plot(array_paneles, resultados_van, marker='o', linestyle='-', color='b')
    plt.axvline(x=n_optimo, color='r', linestyle='--', label=f'Óptimo: {n_optimo} paneles')
    plt.title(f'Curva de optimización: Valor Actual Neto a {DURACION_TOTAL} años')
    plt.xlabel('Número de Paneles')
    plt.ylabel('Valor Actual Neto (VAN) en €')
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.show()

# Vectores de Serie Temporal     
if 'precio_fijo' in globals() and precio_fijo is not None:
    precio_compra_horario = np.full(8760, precio_fijo)
else:
    precio_compra_horario = generar_array_tres_zonas(precio_valle, precio_llano, precio_punta, año_base)
precio_venta_horario = np.full(8760, precio_venta)  


# --- Salida Principal Solicitada ---
def mostrar_resultados_fijos():
    van, pct_autoconsumo, pct_autosuficiencia, flujos, soc_array_ano1, capacidades_bateria = calcular_simulacion_fija(N_PANELES_FIJO, CAPACIDAD_TOTAL_BATERIA_KWH, SOC_MIN_BATERIA_PCT, consumo_base_horaria, gen_base_horaria)
    
    # 1. Output en Terminal
    print(f"\n--- CARACTERÍSTICAS DEL SISTEMA ---")
    print(f"Potencia FV instalada: {N_PANELES_FIJO*POTENCIA_PANEL_W/1000} kW")
    print(f"Capacidad de Batería Física: {CAPACIDAD_TOTAL_BATERIA_KWH} kWh ({N_BATERIAS} x {CAPACIDAD_BATERIA_KWH} kWh)")
    print(f"Límite de Descarga: {SOC_MIN_BATERIA_PCT}%")
    print(f"\n--- RESULTADOS ---")
    print(f"Porcentaje de Autoconsumo: {pct_autoconsumo:.2f}%")
    print(f"Porcentaje de Autosuficiencia: {pct_autosuficiencia:.2f}%")
    print(f"Valor Actual Neto (VAN): {van:.2f} €\n")
    
    # 2. Primera figura: Gráfico de Ahorro Acumulado
    ano_recuperacion = next((i for i, v in enumerate(flujos) if v >= 0), None)

    plt.figure(figsize=(10, 6))
    plt.axhline(y=0, color='black', linestyle='-', zorder=1)
    plt.plot(range(DURACION_TOTAL + 1), flujos, marker='o', markersize=4, linestyle='-', color='g', label='Flujo Acumulado Descontado', zorder=2)
    plt.axvline(x=GARANTIA_PANELES_ANOS, color='r', linestyle='--', label=f'Garantía Paneles ({GARANTIA_PANELES_ANOS} años)')
    if ano_recuperacion is not None:
        plt.axvline(x=ano_recuperacion, color='b', linestyle='--', label=f'Punto de Recuperación (Año {ano_recuperacion})')
    potencia_kw = N_PANELES_FIJO * POTENCIA_PANEL_W / 1000.0
    plt.title(f'Proyección de Ahorro Acumulado ({potencia_kw} kW + Batería {CAPACIDAD_TOTAL_BATERIA_KWH} kWh)\nVAN: {van:,.2f} €')
    plt.xlabel('Años de Operación')
    plt.ylabel('Euros (€)')
    plt.xlim(0, DURACION_TOTAL)
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    #plt.savefig(f"{N_BATERIAS}_Ahorro.png", dpi=300, bbox_inches='tight')
    
    # 3. Segunda figura: Gráfico de Carga Máxima y Mínima Diaria
    if CAPACIDAD_TOTAL_BATERIA_KWH > 0:
        soc_pct = (soc_array_ano1 / CAPACIDAD_TOTAL_BATERIA_KWH) * 100
    else:
        soc_pct = np.zeros(8760)
        
    soc_diario = soc_pct.reshape(365, 24)
    carga_maxima_diaria = soc_diario.max(axis=1)
    carga_minima_diaria = soc_diario.min(axis=1)

    plt.figure(figsize=(10, 6))
    dias = np.arange(1, 366)
    
    plt.axhline(y=0, color='black', linestyle='-', zorder=1)
    plt.plot(dias, carga_maxima_diaria, color='blue', linestyle='-', linewidth=1.5, label='Carga Máxima Diaria (%)', zorder=2)
    plt.plot(dias, carga_minima_diaria, color='red', linestyle='-', linewidth=1.5, label='Carga Minima Diaria (%)', zorder=2)
    plt.axhline(y=SOC_MIN_BATERIA_PCT, color='orange', linestyle='--', linewidth=1, label=f'Límite de Descarga ({SOC_MIN_BATERIA_PCT}%)')
    
    plt.title(f'Evolución de Carga Diaria ({potencia_kw} kW + Batería {CAPACIDAD_TOTAL_BATERIA_KWH} kWh) \nAutoconsumo: {pct_autoconsumo:.0f}% | Autosuficiencia: {pct_autosuficiencia:.0f}%')
    plt.xlabel('Día del Año (1 - 365)')
    plt.ylabel('Carga de la Batería (%)')
    plt.xlim(1, 365)
    plt.ylim(-1, 101) 
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    #plt.savefig(f"{N_BATERIAS}_Carga.png", dpi=300, bbox_inches='tight')
    
    # 4. Tercera figura: Gráfico de Barras de Capacidad de la Batería
    if CAPACIDAD_TOTAL_BATERIA_KWH > 0:
        plt.figure(figsize=(10, 6))
        anos = np.arange(0, DURACION_TOTAL + 1)
        capacidades_totales = np.insert(capacidades_bateria, 0, CAPACIDAD_TOTAL_BATERIA_KWH)
        
        bar_container = plt.bar(anos, capacidades_totales, color='orange', edgecolor='darkorange', alpha=0.8, label='Capacidad de la Batería (kWh)')
        
        handles = [bar_container]
        labels = ['Capacidad de la Batería (kWh)']

        ano_muerte = next((i + 1 for i, v in enumerate(capacidades_bateria) if v == 0), None)
        if ano_muerte is not None:
            line = plt.axvline(x=ano_muerte, color='r', linestyle='--', label=f'Muerte Técnica (Año {ano_muerte})')
            handles.append(line)
            labels.append(f'Muerte Técnica (Año {ano_muerte})')

        plt.title(f'Capacidad de la Batería a lo largo del Proyecto \nCapacidad Inicial: {CAPACIDAD_TOTAL_BATERIA_KWH:.2f} kWh')
        plt.xlabel('Año del Proyecto')
        plt.ylabel('Capacidad de Batería (kWh)')
        plt.xlim(-0.8, DURACION_TOTAL + 0.8)
        plt.ylim(0, CAPACIDAD_TOTAL_BATERIA_KWH)
        plt.grid(True, alpha=0.5, axis='y')
        plt.legend(handles=handles, labels=labels)
        plt.tight_layout()
        #plt.savefig(f"{N_BATERIAS}_Capacidad.png", dpi=300, bbox_inches='tight')

    plt.show()

mostrar_resultados_fijos()
# optimizar_sistema()
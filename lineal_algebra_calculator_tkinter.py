# ---------------------------------------------------------------------------
# lineal_algebra_calculator_tkinter.py
# Versión Optimizada y Refactorizada
# ---------------------------------------------------------------------------
import math
import queue
import threading
import time
import tkinter as tk
import traceback
from tkinter import ttk, scrolledtext, messagebox, font as tkFont, filedialog, StringVar
from tkinter.scrolledtext import ScrolledText
from typing import Any, Callable, Dict, Tuple, Optional, List, Union  # Tipos más específicos

import numpy as np



# Configuración inicial de impresión de NumPy (se podrá ajustar desde la GUI)
DEFAULT_PRINT_PRECISION: int = 6
np.set_printoptions(precision=DEFAULT_PRINT_PRECISION, suppress=True, linewidth=160)

# Definir tolerancias pequeñas para comparaciones globales
ZERO_TOLERANCE: float = 1e-15
SMALL_PIVOT_WARNING_TOLERANCE: float = 1e-9

# Límite para la creación de widgets de entrada en la GUI
MAX_N_FOR_GUI_ENTRIES: int = 20


# ===========================================================================
# MÉTODOS NUMÉRICOS Y FUNCIONES AUXILIARES
# Modificados para devolver un único diccionario 'details'
# ===========================================================================

def es_simetrica_burden(A_matrix: np.ndarray, tol: float = 1e-8) -> bool:
    if not isinstance(A_matrix, np.ndarray) or A_matrix.ndim != 2 or A_matrix.shape[0] != A_matrix.shape[1]:
        return False
    return np.allclose(A_matrix, A_matrix.T, atol=tol)


def es_definida_positiva_burden(A_matrix: np.ndarray, sym_tol: float = 1e-8, eig_tol: float = 1e-12) -> bool:
    if not es_simetrica_burden(A_matrix, tol=sym_tol):
        return False
    try:
        eigenvalues = np.linalg.eigvalsh(A_matrix)
        return np.all(eigenvalues > eig_tol)
    except np.linalg.LinAlgError:
        return False


def check_diagonal_dominance_burden(A_matrix: np.ndarray) -> Dict[str, Any]:
    details: Dict[str, Any] = {"steps_detailed": [], "status": ""}
    if not isinstance(A_matrix, np.ndarray) or A_matrix.ndim != 2 or A_matrix.shape[0] != A_matrix.shape[1]:
        details["status"] = "Error: La entrada no es una matriz NumPy cuadrada válida."
        return details

    A = A_matrix
    N = A.shape[0]
    details_list: List[str] = []  # Renombrado para evitar conflicto con el dict 'details'
    count_strictly_dominant_rows = 0
    count_weakly_compliant_rows = 0
    strict_comparison_epsilon = 1e-9
    equality_atol = 1e-12

    for i_py in range(N):
        i_burden = i_py + 1
        abs_aii = np.abs(A[i_py, i_py])
        sum_abs_aij_off_diag = 0.0
        for j_py in range(N):
            if i_py != j_py: sum_abs_aij_off_diag += np.abs(A[i_py, j_py])

        part1_fila_info = f"Fila {i_burden}: |a_{{{i_burden},{i_burden}}}| = {abs_aii:.4e}, "
        sum_intro_str = f"Σ_{{j≠{i_burden}}}"
        sum_term_str = f"|a_{{{i_burden}}},j|"
        sum_val_formatted = f"{sum_abs_aij_off_diag:.4e}"
        part2_suma_info = f"{sum_intro_str} {sum_term_str} = {sum_val_formatted}. "
        detail_prefix = part1_fila_info + part2_suma_info

        if abs_aii > sum_abs_aij_off_diag + strict_comparison_epsilon:
            details_list.append(detail_prefix + "Estrictamente dominante (|a_ii| > Σ|a_ij|, j≠i).")
            count_strictly_dominant_rows += 1
            count_weakly_compliant_rows += 1
        elif np.isclose(abs_aii, sum_abs_aij_off_diag, atol=equality_atol, rtol=0):
            details_list.append(detail_prefix + "Débilmente dominante (cumple |a_ii| ≈ Σ|a_ij|, j≠i).")
            count_weakly_compliant_rows += 1
        elif abs_aii >= sum_abs_aij_off_diag:
            details_list.append(
                detail_prefix + "Débilmente dominante (|a_ii| >= Σ|a_ij|, j≠i, pero no estricta por epsilon o igualdad).")
            count_weakly_compliant_rows += 1
        else:
            details_list.append(detail_prefix + "NO dominante (|a_ii| < Σ|a_ij|, j≠i).")

    summary = ""
    if count_weakly_compliant_rows < N:
        summary = "La matriz NO es diagonalmente dominante."
    elif count_strictly_dominant_rows == N:
        summary = "La matriz es ESTRICTAMENTE DIAGONALMENTE DOMINANTE."
    elif count_weakly_compliant_rows == N and count_strictly_dominant_rows > 0:
        summary = "La matriz es DÉBILMENTE DIAGONALMENTE DOMINANTE (con al menos una fila estricta)."
    elif count_weakly_compliant_rows == N and count_strictly_dominant_rows == 0:
        summary = "La matriz es DÉBILMENTE DIAGONALMENTE DOMINANTE (ninguna fila estrictamente dominante)."
    else:
        summary = f"Análisis de dominancia con resultado inesperado (Débiles: {count_weakly_compliant_rows}, Estrictas: {count_strictly_dominant_rows})."

    details["status"] = summary
    details["steps_detailed"] = details_list  # Asignar la lista de strings a la clave correcta
    details["details_text"] = "\n".join(details_list)  # Para display_output
    return details


def _forward_substitution(L_matrix: np.ndarray, c_vector: np.ndarray, details_list: List[str],
                          matrix_name_L: str = "L", vector_name_c: str = "c",
                          result_vector_name: str = "Y") -> Tuple[Optional[np.ndarray], str]:
    N = L_matrix.shape[0]
    Y_solution = np.zeros(N)
    details_list.append(
        f"--- Resolviendo {matrix_name_L}{result_vector_name} = {vector_name_c} (Sustitución Hacia Adelante) ---")
    for i_py in range(N):
        i_burden = i_py + 1
        sum_LY_val = 0.0
        # sum_LY_str_terms = [] # No se usa fuera de esta iteración, se puede simplificar
        for j_py in range(i_py):
            term = L_matrix[i_py, j_py] * Y_solution[j_py]
            sum_LY_val += term
            # sum_LY_str_terms.append(f"{matrix_name_L}[{i_burden},{j_py + 1}]*{result_vector_name}[{j_py + 1}]({L_matrix[i_py, j_py]:.4e}*{Y_solution[j_py]:.4e})")
        diag_Lii = L_matrix[i_py, i_py]
        if np.isclose(diag_Lii, 0.0, atol=ZERO_TOLERANCE):
            status_msg = f"Error Crítico: {matrix_name_L}[{i_burden},{i_burden}]={diag_Lii:.2e} es (cercano a) cero (tol={ZERO_TOLERANCE:.1e})."
            details_list.append(status_msg)
            return None, status_msg
        elif abs(diag_Lii) < SMALL_PIVOT_WARNING_TOLERANCE and not np.isclose(diag_Lii, 0.0, atol=ZERO_TOLERANCE):
            details_list.append(
                f"Advertencia: {matrix_name_L}[{i_burden},{i_burden}]={diag_Lii:.2e} muy pequeño (<{SMALL_PIVOT_WARNING_TOLERANCE:.1e}).")

        Y_solution[i_py] = (c_vector[i_py] - sum_LY_val) / diag_Lii
        # sum_LY_str_display = " + ".join(sum_LY_str_terms) if sum_LY_str_terms else "0.0" # Simplificado
        details_list.append(
            f"{result_vector_name}[{i_burden}] = ({vector_name_c}[{i_burden}]({c_vector[i_py]:.4e}) - Suma({sum_LY_val:.4e})) / {matrix_name_L}[{i_burden},{i_burden}]({diag_Lii:.4e}) = {Y_solution[i_py]:.6e}")
    details_list.append(f"Vector {result_vector_name}:\n{Y_solution}\n")
    return Y_solution, "OK"


def _backward_substitution(U_matrix: np.ndarray, Y_vector: np.ndarray, details_list: List[str],
                           matrix_name_U: str = "U", vector_name_Y: str = "Y",
                           result_vector_name: str = "X") -> Tuple[Optional[np.ndarray], str]:
    N = U_matrix.shape[0]
    X_solution = np.zeros(N)
    details_list.append(
        f"--- Resolviendo {matrix_name_U}{result_vector_name} = {vector_name_Y} (Sustitución Hacia Atrás) ---")
    for i_py in range(N - 1, -1, -1):
        i_burden = i_py + 1
        sum_UX_val = 0.0
        # sum_UX_str_terms = [] # No se usa fuera
        for j_py in range(i_py + 1, N):
            term = U_matrix[i_py, j_py] * X_solution[j_py]
            sum_UX_val += term
            # sum_UX_str_terms.append(f"{matrix_name_U}[{i_burden},{j_py + 1}]*{result_vector_name}[{j_py + 1}]({U_matrix[i_py, j_py]:.4f}*{X_solution[j_py]:.4f})")
        diag_Uii = U_matrix[i_py, i_py]
        if np.isclose(diag_Uii, 0.0, atol=ZERO_TOLERANCE):
            status_msg = f"Error Crítico: {matrix_name_U}[{i_burden},{i_burden}]={diag_Uii:.2e} es (cercano a) cero (tol={ZERO_TOLERANCE:.1e})."
            details_list.append(status_msg)
            return None, status_msg
        elif abs(diag_Uii) < SMALL_PIVOT_WARNING_TOLERANCE and not np.isclose(diag_Uii, 0.0, atol=ZERO_TOLERANCE):
            details_list.append(
                f"Advertencia: {matrix_name_U}[{i_burden},{i_burden}]={diag_Uii:.2e} muy pequeño (<{SMALL_PIVOT_WARNING_TOLERANCE:.1e}).")

        X_solution[i_py] = (Y_vector[i_py] - sum_UX_val) / diag_Uii
        # sum_UX_str_display = " + ".join(sum_UX_str_terms) if sum_UX_str_terms else "0.0" # Simplificado
        details_list.append(
            f"{result_vector_name}[{i_burden}] = ({vector_name_Y}[{i_burden}]({Y_vector[i_py]:.4e}) - Suma({sum_UX_val:.4e})) / {matrix_name_U}[{i_burden},{i_burden}]({diag_Uii:.4e}) = {X_solution[i_py]:.6e}")
    details_list.append(f"Vector {result_vector_name}:\n{X_solution}\n")
    return X_solution, "OK"


# ... (Resto de tus funciones numéricas: calculate_determinant_burden, calculate_inverse_burden, etc.
#      DEBEN SER MODIFICADAS para que el resultado principal (determinante, inversa, L, U, etc.)
#      se guarde en el diccionario 'details' y la función devuelva *solo* ese diccionario 'details'.
#      Ejemplo para calculate_determinant_burden:)

def calculate_determinant_burden(A_matrix: np.ndarray) -> Dict[str, Any]:
    details: Dict[str, Any] = {"steps_detailed": [], "status": "Iniciando cálculo de determinante."}
    try:
        if A_matrix.shape[0] != A_matrix.shape[1]:
            details["status"] = "Error: La matriz debe ser cuadrada."
            return details  # Devuelve solo details
        det = np.linalg.det(A_matrix)
        details["status"] = "Determinante calculado."
        details["determinant_value"] = det  # Guardar el valor en details
        details["steps_detailed"].append(f"Usando numpy.linalg.det(A).\ndet(A) (valor crudo): {det}")
        if np.isclose(det, 0, atol=ZERO_TOLERANCE):
            details["steps_detailed"].append(
                f"Advertencia: Determinante ({det:.2e}) cercano a cero (tol={ZERO_TOLERANCE:.1e}). Matriz singular o casi singular.")
        return details  # Devuelve solo details
    except Exception as e:
        details["status"] = f"Error al calcular determinante: {str(e)}"
        return details


# --- (Aplica un patrón similar a TODAS las funciones numéricas principales) ---

# --- Workers específicos para métodos de solución que usan factorizaciones ---
def _solve_lu_burden_worker(L_matrix: np.ndarray, U_matrix: np.ndarray,
                            P_factor: Optional[np.ndarray], b_vector_orig: np.ndarray) -> Dict[str, Any]:
    details: Dict[str, Any] = {"status": "Procesando Solución Ax=b usando LU", "steps_detailed": []}
    # ... (lógica de solve_lu_burden, adaptada para poner X_solution en details) ...
    # Esta función interna ya existe como solve_lu_burden, solo asegúrate que devuelva el dict details
    X_solution, details_solve = solve_lu_burden(L_matrix, U_matrix, P_factor,
                                                b_vector_orig)  # Asume que solve_lu_burden ya devuelve (X, details)
    if X_solution is not None:
        details_solve["X_solution"] = X_solution  # Asegurar que esté en el dict
    return details_solve


def _cholesky_solve_burden_worker(L_chol_matrix: np.ndarray, b_vector_orig: np.ndarray) -> Dict[str, Any]:
    details: Dict[str, Any] = {"status": "Procesando Solución vía Cholesky", "steps_detailed": [], "tag": "info"}
    details["steps_detailed"].append("Resolviendo Ax=b donde A=LL^T. Primero Ly=b, luego L^T X = Y.")
    Y_vec, status_fw = _forward_substitution(L_chol_matrix, b_vector_orig, details["steps_detailed"],
                                             matrix_name_L="L_chol", vector_name_c="b", result_vector_name="Y")
    solution_X_cs = None
    if Y_vec is None:
        details["status"] = status_fw;
        details["tag"] = "error"
    else:
        solution_X_cs, status_bw = _backward_substitution(L_chol_matrix.T, Y_vec, details["steps_detailed"],
                                                          matrix_name_U="L_chol^T", vector_name_Y="Y",
                                                          result_vector_name="X")
        if solution_X_cs is None:
            details["status"] = status_bw;
            details["tag"] = "error"
        else:
            details["status"] = "Solución (vía Cholesky) encontrada exitosamente."
            details["tag"] = "success"
            details["X_solution"] = solution_X_cs  # Guardar la solución en details
    return details


class NumericalAnalysisApp:
    # ... (atributos de clase como MAX_N_FOR_GUI_ENTRIES) ...

    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        master.title("Calculadora de Álgebra Lineal Numérica Avanzada (Notación Burden & Faires)")
        master.geometry("1250x900")

        self.style = ttk.Style()
        available_themes = self.style.theme_names()
        if "vista" in available_themes:
            self.style.theme_use('vista')
        elif "clam" in available_themes:
            self.style.theme_use('clam')
        elif available_themes:
            self.style.theme_use(available_themes[0])

        self.configure_styles()

        self.A_matrix: Optional[np.ndarray] = None
        self.b_vector: Optional[np.ndarray] = None
        self.x0_vector: Optional[np.ndarray] = None
        self.L_matrix_factor: Optional[np.ndarray] = None
        self.U_matrix_factor: Optional[np.ndarray] = None
        self.P_factor: Optional[np.ndarray] = None  # Puede ser NROW (vector) o Matriz P
        self.L_chol_matrix: Optional[np.ndarray] = None
        self.A_transformed_by_GE: Optional[np.ndarray] = None
        self.b_transformed_by_GE: Optional[np.ndarray] = None
        self.NROW_from_GE: Optional[List[int]] = None
        self.last_iteration_history: List[Tuple[int, np.ndarray, float]] = []
        self.last_iter_params: Dict[str, Any] = {}

        self.n_var = tk.StringVar(value="3")
        self.show_detailed_steps_var = tk.BooleanVar(value=True)
        self.show_iter_history_in_gui_var = tk.BooleanVar(value=True)
        self.decimal_precision_var = tk.StringVar(value=str(DEFAULT_PRINT_PRECISION))
        self.current_display_precision: int = DEFAULT_PRINT_PRECISION

        self.result_queue: queue.Queue = queue.Queue()
        self.calculation_thread: Optional[threading.Thread] = None
        self.is_calculating: bool = False
        self.buttons_to_disable_during_calc: List[tk.Widget] = []

        # --- UI Principal ---
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_controls_frame = ttk.Frame(main_frame)
        left_controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5, anchor=tk.N)

        input_panel = ttk.LabelFrame(left_controls_frame, text="Entrada de Datos", padding="10")
        input_panel.pack(side=tk.TOP, fill=tk.X, padx=0, pady=0, ipady=5)

        n_frame = ttk.Frame(input_panel)
        n_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(n_frame, text="Dimensión N:").pack(side=tk.LEFT, padx=(0, 2))
        self.n_entry = ttk.Entry(n_frame, textvariable=self.n_var, width=5)
        self.n_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.n_entry.bind("<Return>", self.create_matrix_entries_event_wrapper)
        self.btn_crear_reiniciar = ttk.Button(n_frame, text="Crear/Reiniciar Entradas",
                                              command=self.create_matrix_entries_event_wrapper, style="Accent.TButton")
        self.btn_crear_reiniciar.pack(side=tk.LEFT, padx=5)
        self.buttons_to_disable_during_calc.append(self.btn_crear_reiniciar)

        self.matrix_entry_frame = ttk.Frame(input_panel)
        self.matrix_entry_frame.pack(fill=tk.X, pady=5)
        self.matrix_A_entries: List[List[ttk.Entry]] = []
        self.vector_b_entries: List[ttk.Entry] = []
        self.vector_x0_entries: List[ttk.Entry] = []

        io_frame = ttk.Frame(input_panel)
        io_frame.pack(fill=tk.X, pady=(5, 5))
        self.btn_cargar_a = ttk.Button(io_frame, text="Cargar A", command=lambda: self.load_data_from_file('A'))
        self.btn_cargar_a.pack(side=tk.LEFT, padx=2);
        self.buttons_to_disable_during_calc.append(self.btn_cargar_a)
        self.btn_cargar_b = ttk.Button(io_frame, text="Cargar b", command=lambda: self.load_data_from_file('b'))
        self.btn_cargar_b.pack(side=tk.LEFT, padx=2);
        self.buttons_to_disable_during_calc.append(self.btn_cargar_b)
        self.btn_cargar_x0 = ttk.Button(io_frame, text="Cargar X(0)", command=lambda: self.load_data_from_file('X0'))
        self.btn_cargar_x0.pack(side=tk.LEFT, padx=2);
        self.buttons_to_disable_during_calc.append(self.btn_cargar_x0)
        self.btn_guardar_a = ttk.Button(io_frame, text="Guardar A", command=lambda: self.save_matrix_to_file('A'))
        self.btn_guardar_a.pack(side=tk.LEFT, padx=2);
        self.buttons_to_disable_during_calc.append(self.btn_guardar_a)

        output_options_panel = ttk.LabelFrame(left_controls_frame, text="Opciones de Salida", padding="10")
        output_options_panel.pack(side=tk.TOP, fill=tk.X, padx=0, pady=10)
        detail_checkbox = ttk.Checkbutton(output_options_panel, text="Mostrar Pasos Detalladísimos en GUI",
                                          variable=self.show_detailed_steps_var)
        detail_checkbox.pack(anchor=tk.W, pady=2)
        iter_hist_checkbox = ttk.Checkbutton(output_options_panel, text="Mostrar Historial de Iteraciones en GUI",
                                             variable=self.show_iter_history_in_gui_var)
        iter_hist_checkbox.pack(anchor=tk.W, pady=2)
        iter_limit_frame = ttk.Frame(output_options_panel)
        iter_limit_frame.pack(anchor=tk.W, pady=2, fill=tk.X)
        ttk.Label(iter_limit_frame, text="Mostrar k iter. (0=todas):").pack(side=tk.LEFT)
        self.iter_display_limit_spinbox = ttk.Spinbox(iter_limit_frame, from_=0, to=200, increment=10, width=4,
                                                      justify='right')
        self.iter_display_limit_spinbox.set("10")
        self.iter_display_limit_spinbox.pack(side=tk.LEFT, padx=(2, 5))
        precision_frame = ttk.Frame(output_options_panel)
        precision_frame.pack(anchor=tk.W, pady=2, fill=tk.X)
        ttk.Label(precision_frame, text="Decimales en Salida:").pack(side=tk.LEFT)
        self.decimal_precision_entry = ttk.Entry(precision_frame, textvariable=self.decimal_precision_var, width=3,
                                                 justify='right')
        self.decimal_precision_entry.pack(side=tk.LEFT, padx=(2, 5))
        self.btn_aplicar_precision = ttk.Button(precision_frame, text="Aplicar Precisión",
                                                command=self.apply_output_settings)
        self.btn_aplicar_precision.pack(side=tk.LEFT);
        self.buttons_to_disable_during_calc.append(self.btn_aplicar_precision)

        methods_panel = ttk.LabelFrame(main_frame, text="Selección de Métodos", padding="10")
        methods_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        notebook = ttk.Notebook(methods_panel)
        direct_methods_tab = ttk.Frame(notebook, padding="5");
        notebook.add(direct_methods_tab, text='Sist. Lineales (Directos)')
        self.create_direct_methods_ui(direct_methods_tab)
        iterative_methods_tab = ttk.Frame(notebook, padding="5");
        notebook.add(iterative_methods_tab, text='Sist. Lineales (Iterativos)')
        self.create_iterative_methods_ui(iterative_methods_tab)
        factorizations_tab = ttk.Frame(notebook, padding="5");
        notebook.add(factorizations_tab, text='Factorizaciones')
        self.create_factorizations_ui(factorizations_tab)
        adv_analysis_tab = ttk.Frame(notebook, padding="5");
        notebook.add(adv_analysis_tab, text='Análisis Avanzado')
        self.create_advanced_analysis_ui(adv_analysis_tab)
        analysis_tab = ttk.Frame(notebook, padding="5");
        notebook.add(analysis_tab, text='Análisis Simple')
        self.create_simple_analysis_ui(analysis_tab)
        notebook.pack(expand=True, fill=tk.BOTH)

        output_panel = ttk.LabelFrame(main_frame, text="Resultados y Pasos", padding="10")
        output_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5, ipadx=5, ipady=5)

        self.output_text = scrolledtext.ScrolledText(output_panel, wrap=tk.WORD, height=25, width=90, relief=tk.SOLID,
                                                     borderwidth=1)
        self.output_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.output_text.configure(font=("Courier New", 9))
        self.output_text.tag_configure('title', font=('Helvetica', 12, 'bold'), foreground="#003366")
        self.output_text.tag_configure('error', foreground="red")
        self.output_text.tag_configure('warning', foreground="orange")
        self.output_text.tag_configure('success', foreground="green")
        self.output_text.tag_configure('info', foreground="blue")
        self.output_text.tag_configure('matrix_label', font=('Helvetica', 10, 'bold'))

        output_controls_bottom_frame = ttk.Frame(output_panel)
        output_controls_bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        self.export_iter_button = ttk.Button(output_controls_bottom_frame, text="Exportar Iteraciones",
                                             command=self.export_iteration_history, state=tk.DISABLED)
        self.export_iter_button.pack(side=tk.LEFT, padx=2);
        self.buttons_to_disable_during_calc.append(self.export_iter_button)
        self.export_full_output_button = ttk.Button(output_controls_bottom_frame, text="Exportar Resultados",
                                                    command=self.export_full_output_to_txt)
        self.export_full_output_button.pack(side=tk.LEFT, padx=2);
        self.buttons_to_disable_during_calc.append(self.export_full_output_button)
        self.btn_limpiar_salida = ttk.Button(output_controls_bottom_frame, text="Limpiar Salida",
                                             command=self.clear_output)
        self.btn_limpiar_salida.pack(side=tk.RIGHT, padx=2);
        self.buttons_to_disable_during_calc.append(self.btn_limpiar_salida)

        progress_status_frame = ttk.Frame(output_panel)
        progress_status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 0))  # Ajuste de pady
        self.progress_bar = ttk.Progressbar(progress_status_frame, orient=tk.HORIZONTAL, length=300,
                                            mode='indeterminate')
        self.status_label = ttk.Label(progress_status_frame, text="")
        # La barra y el label se empaquetan/desempaquetan dinámicamente

        self.create_matrix_entries_event_wrapper()
        self.apply_output_settings()
        self.master.after(100, self.process_result_queue)

    def configure_styles(self):
        self.style.configure("TLabel", padding=2, font=('Helvetica', 10))
        self.style.configure("TButton", padding=5, font=('Helvetica', 10))
        self.style.configure("Accent.TButton", foreground="white", background="#0078D7", font=('Helvetica', 10, 'bold'))
        self.style.map("Accent.TButton", background=[('active', '#005A9E')],
                       relief=[('pressed', 'sunken'), ('!pressed', 'raised')])
        self.style.configure("TEntry", padding=3, font=('Helvetica', 10), fieldbackground='white')
        self.style.configure("TLabelframe.Label", font=('Helvetica', 11, 'bold'), padding=(0, 0, 0, 5))
        self.style.configure("TNotebook.Tab", font=('Helvetica', 10, 'bold'), padding=(5, 2))
        self.style.configure("TSpinbox", padding=3, font=('Helvetica', 10))

    def _toggle_calculation_buttons(self, enable: bool):
        new_state = tk.NORMAL if enable else tk.DISABLED
        for btn_widget in self.buttons_to_disable_during_calc:
            try:
                # OptionMenu se maneja diferente para el estado
                if isinstance(btn_widget, ttk.OptionMenu):
                    btn_widget.config(state=tk.ACTIVE if enable else tk.DISABLED)
                else:
                    btn_widget.config(state=new_state)
            except tk.TclError:
                # Algunos widgets podrían no tener 'state', o ya estar en el estado deseado
                pass

    def start_calculation_feedback(self, message: str = "Calculando..."):
        if self.is_calculating:  # Prevenir múltiples inicios si ya está calculando
            return
        self.is_calculating = True
        self._toggle_calculation_buttons(False)

        self.status_label.config(text=message)
        # Empaquetar solo si no están ya empaquetados para evitar errores
        if not self.status_label.winfo_ismapped():
            self.status_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        if not self.progress_bar.winfo_ismapped():
            self.progress_bar.pack(side=tk.LEFT, padx=5, pady=2, fill=tk.X, expand=True)

        self.progress_bar.start(15)
        self.master.update_idletasks()

    def stop_calculation_feedback(self):
        self.progress_bar.stop()
        if self.progress_bar.winfo_ismapped():
            self.progress_bar.pack_forget()
        if self.status_label.winfo_ismapped():
            self.status_label.pack_forget()
        self.status_label.config(text="")
        self.is_calculating = False
        self._toggle_calculation_buttons(True)
        self.master.update_idletasks()

    def process_result_queue(self):
        try:
            result_package: Dict[str, Any] = self.result_queue.get_nowait()

            method_name = result_package.get("method_name", "Operación")
            exec_time = result_package.get("time", 0.0)

            self.stop_calculation_feedback()

            if result_package["type"] == "result":
                data_to_display = result_package["data"]
                self.display_output(f"Resultado: {method_name}", data_to_display)
            elif result_package["type"] == "error":
                error_type_name = result_package.get("error_type", "ErrorDesconocido")
                message = result_package.get("message", "Error desconocido en el hilo.")
                tb_str = result_package.get("traceback", "No hay traceback disponible.")
                messagebox.showerror(f"Error en {method_name}",
                                     f"Ocurrió un error durante el cálculo:\n{error_type_name}: {message}")
                self.display_output(f"Excepción en {method_name}",
                                    {"status": f"{error_type_name}: {message}\n\nTraceback:\n{tb_str}", "tag": "error"})

            self.output_text.configure(state='normal')
            self.output_text.insert(tk.END,
                                    f"\n--- Tiempo de ejecución (hilo) de '{method_name}': {exec_time:.6f} segundos. ---\n",
                                    ('info',))
            self.output_text.configure(state='disabled')
            self.output_text.see(tk.END)

        except queue.Empty:
            pass
        finally:
            self.master.after(150, self.process_result_queue)  # Ajustar el tiempo de sondeo si es necesario

    def calculation_worker(self, method_name: str, target_function: Callable, args: Tuple, kwargs: Dict[str, Any]):
        start_time = time.time()
        try:
            details_from_function = target_function(*args, **kwargs)  # Se espera que devuelva un dict

            if not isinstance(details_from_function, dict):
                # Si la función no sigue la convención, crear un dict de error/advertencia
                print(
                    f"ADVERTENCIA: La función {target_function.__name__} para '{method_name}' no devolvió un diccionario como se esperaba.")
                final_details_dict = {
                    "status": f"Error interno: {target_function.__name__} no devolvió un diccionario de detalles.",
                    "raw_output": str(details_from_function),
                    "tag": "error"
                }
            else:
                final_details_dict = details_from_function

            package = {
                "type": "result",
                "method_name": method_name,
                "data": final_details_dict,
                "time": time.time() - start_time
            }
            self.result_queue.put(package)

        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"EXCEPCIÓN en hilo para {method_name}:\n{tb_str}")
            self.result_queue.put({
                "type": "error",
                "method_name": method_name,
                "error_type": type(e).__name__,
                "message": str(e),
                "traceback": tb_str,
                "time": time.time() - start_time
            })

    # --- Métodos de creación de UI para pestañas (Asegúrate que los botones se añadan a self.buttons_to_disable_during_calc) ---
    def create_direct_methods_ui(self, tab: ttk.Frame):
        self.use_scaled_pivoting_var = tk.BooleanVar(value=True)
        piv_check = ttk.Checkbutton(tab, text="Usar Pivoteo Parcial Escalonado (sino Simple) en Gaussiana",
                                    variable=self.use_scaled_pivoting_var)
        piv_check.pack(pady=(0, 5), anchor=tk.W)

        btn1 = ttk.Button(tab, text="Eliminación Gaussiana (Algs. 6.1, 6.2/6.3)", style="Accent.TButton",
                          command=lambda: self.execute_method("Eliminación Gaussiana", gaussian_elimination_burden))
        btn1.pack(pady=3, fill=tk.X);
        self.buttons_to_disable_during_calc.append(btn1)

        btn2 = ttk.Button(tab, text="Eliminación Gauss-Jordan",
                          command=lambda: self.execute_method("Gauss-Jordan", gauss_jordan_elimination_burden))
        btn2.pack(pady=3, fill=tk.X);
        self.buttons_to_disable_during_calc.append(btn2)

        btn3 = ttk.Button(tab, text="Resolver con LU (Factorización Actual)",
                          command=lambda: self.execute_method("LU Solve", _solve_lu_burden_worker,
                                                              is_factorization=False))
        btn3.pack(pady=3, fill=tk.X);
        self.buttons_to_disable_during_calc.append(btn3)

        btn4 = ttk.Button(tab, text="Resolver con Cholesky (Factorización Actual)",
                          command=lambda: self.execute_method("Cholesky Solve", _cholesky_solve_burden_worker,
                                                              is_factorization=False))
        btn4.pack(pady=3, fill=tk.X);
        self.buttons_to_disable_during_calc.append(btn4)

    def create_iterative_methods_ui(self, tab: ttk.Frame):
        params_frame = ttk.Frame(tab);
        params_frame.pack(pady=5, fill=tk.X)
        ttk.Label(params_frame, text="TOL:").pack(side=tk.LEFT, padx=(0, 2))
        self.tol_entry = ttk.Entry(params_frame, width=10, justify='right');
        self.tol_entry.insert(0, "1e-8");
        self.tol_entry.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(params_frame, text="NMAX:").pack(side=tk.LEFT, padx=(0, 2))
        self.n_iter_entry = ttk.Entry(params_frame, width=7, justify='right');
        self.n_iter_entry.insert(0, "100");
        self.n_iter_entry.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(params_frame, text="ω (SOR):").pack(side=tk.LEFT, padx=(0, 2))
        self.omega_entry = ttk.Entry(params_frame, width=5, justify='right');
        self.omega_entry.insert(0, "1.2");
        self.omega_entry.pack(side=tk.LEFT, padx=(0, 10))

        btn1 = ttk.Button(tab, text="Método de Jacobi (Alg. 7.1)", style="Accent.TButton",
                          command=lambda: self.execute_method("Jacobi", jacobi_method_burden, requires_X0=True))
        btn1.pack(pady=3, fill=tk.X);
        self.buttons_to_disable_during_calc.append(btn1)
        btn2 = ttk.Button(tab, text="Método de Gauss-Seidel (Alg. 7.2)",
                          command=lambda: self.execute_method("Gauss-Seidel", gauss_seidel_method_burden,
                                                              requires_X0=True))
        btn2.pack(pady=3, fill=tk.X);
        self.buttons_to_disable_during_calc.append(btn2)
        btn3 = ttk.Button(tab, text="Método SOR (Alg. 7.3)",
                          command=lambda: self.execute_method("SOR", sor_method_burden, requires_X0=True,
                                                              requires_omega=True))
        btn3.pack(pady=3, fill=tk.X);
        self.buttons_to_disable_during_calc.append(btn3)

    def create_factorizations_ui(self, tab: ttk.Frame):
        btn1 = ttk.Button(tab, text="Factorización LU (Doolittle: l_ii=1, Alg. 6.4)", style="Accent.TButton",
                          command=lambda: self.execute_method("Doolittle (LU)", lu_factorization_doolittle_burden,
                                                              requires_b=False, is_factorization=True))
        btn1.pack(pady=3, fill=tk.X);
        self.buttons_to_disable_during_calc.append(btn1)
        btn2 = ttk.Button(tab, text="Factorización LU (Crout: u_ii=1, Alg. 6.4 adaptado)",
                          command=lambda: self.execute_method("Crout (LU)", lu_factorization_crout_burden,
                                                              requires_b=False, is_factorization=True))
        btn2.pack(pady=3, fill=tk.X);
        self.buttons_to_disable_during_calc.append(btn2)
        btn3 = ttk.Button(tab, text="Factorización de Cholesky (A=LL^T, Alg. 6.6)",
                          command=lambda: self.execute_method("Cholesky", cholesky_factorization_burden,
                                                              requires_b=False, is_factorization=True))
        btn3.pack(pady=3, fill=tk.X);
        self.buttons_to_disable_during_calc.append(btn3)

    def create_simple_analysis_ui(self, tab: ttk.Frame):
        btn1 = ttk.Button(tab, text="Verificar Dominancia Diagonal (Def. 7.21)", style="Accent.TButton",
                          command=lambda: self.execute_method("Dominancia Diagonal", check_diagonal_dominance_burden,
                                                              requires_b=False, is_analysis=True))
        btn1.pack(pady=3, fill=tk.X);
        self.buttons_to_disable_during_calc.append(btn1)
        btn2 = ttk.Button(tab, text="Verificar Simetría",
                          command=lambda: self.execute_method_analysis_simple("Simetría", es_simetrica_burden))
        btn2.pack(pady=3, fill=tk.X);
        self.buttons_to_disable_during_calc.append(btn2)
        btn3 = ttk.Button(tab, text="Verificar Definida Positiva (Def. 6.24)",
                          command=lambda: self.execute_method_analysis_simple("Definida Positiva",
                                                                              es_definida_positiva_burden))
        btn3.pack(pady=3, fill=tk.X);
        self.buttons_to_disable_during_calc.append(btn3)

    def create_advanced_analysis_ui(self, tab: ttk.Frame):
        btn1 = ttk.Button(tab, text="Determinante de A", style="Accent.TButton",
                          command=lambda: self.execute_matrix_operation("Determinante", calculate_determinant_burden))
        btn1.pack(pady=3, fill=tk.X);
        self.buttons_to_disable_during_calc.append(btn1)
        btn2 = ttk.Button(tab, text="Inversa de A (A^-1)",
                          command=lambda: self.execute_matrix_operation("Inversa", calculate_inverse_burden))
        btn2.pack(pady=3, fill=tk.X);
        self.buttons_to_disable_during_calc.append(btn2)
        btn3 = ttk.Button(tab, text="Autovalores y Autovectores de A",
                          command=lambda: self.execute_matrix_operation("Autovalores/vectores",
                                                                        calculate_eigenvalues_vectors_burden))
        btn3.pack(pady=3, fill=tk.X);
        self.buttons_to_disable_during_calc.append(btn3)

        norms_frame = ttk.LabelFrame(tab, text="Normas", padding=5);
        norms_frame.pack(pady=(5, 2), fill=tk.X)
        btn_norm = ttk.Button(norms_frame, text="Normas de A y b",
                              command=lambda: self.execute_matrix_operation("Normas", calculate_matrix_norms_burden,
                                                                            requires_b_optional=True))
        btn_norm.pack(pady=3, fill=tk.X);
        self.buttons_to_disable_during_calc.append(btn_norm)

        cond_frame = ttk.LabelFrame(tab, text="Número de Condición de A", padding=5);
        cond_frame.pack(pady=(5, 2), fill=tk.X)
        self.cond_norm_var = tk.StringVar(value="2")
        cond_options_frame = ttk.Frame(cond_frame);
        cond_options_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(cond_options_frame, text="Usar Norma:").pack(side=tk.LEFT, padx=(0, 5))
        self.cond_norm_menu = ttk.OptionMenu(cond_options_frame, self.cond_norm_var, "2", "1", "2", "inf", "fro", "-1",
                                             "-2", "-inf")
        self.cond_norm_menu.pack(side=tk.LEFT, padx=(0, 10));
        self.buttons_to_disable_during_calc.append(self.cond_norm_menu)
        btn_cond = ttk.Button(cond_frame, text="Calcular Cond(A)",
                              command=lambda: self.execute_matrix_operation("Número de Condición",
                                                                            calculate_condition_number_burden,
                                                                            pass_norm_type=True))
        btn_cond.pack(pady=3, fill=tk.X, side=tk.BOTTOM);
        self.buttons_to_disable_during_calc.append(btn_cond)

        decomp_frame = ttk.LabelFrame(tab, text="Otras Descomposiciones", padding=5);
        decomp_frame.pack(pady=(5, 2), fill=tk.X)
        btn_qr = ttk.Button(decomp_frame, text="Descomposición QR (A=QR)",
                            command=lambda: self.execute_matrix_operation("Descomposición QR", decomposition_qr_burden))
        btn_qr.pack(pady=3, fill=tk.X);
        self.buttons_to_disable_during_calc.append(btn_qr)
        btn_svd = ttk.Button(decomp_frame, text="Descomposición SVD (A=USV^H)",
                             command=lambda: self.execute_matrix_operation("Descomposición SVD",
                                                                           decomposition_svd_burden))
        btn_svd.pack(pady=3, fill=tk.X);
        self.buttons_to_disable_during_calc.append(btn_svd)

    # --- (Resto de los métodos de la clase: _create_matrix_entries, get_matrix_from_entries, etc.) ---
    # --- (Asegúrate que estén actualizados con las últimas versiones que hemos discutido) ---
    # --- (Incluyendo export_full_output_to_txt, y la lógica de X0 nulo en get_vector_from_entries) ---
    # --- (Y los execute_method / execute_matrix_operation adaptados para threading) ---

    def create_matrix_entries_event_wrapper(self, event=None):
        if self.is_calculating: return
        try:
            n_val = int(self.n_var.get())
            if n_val <= MAX_N_FOR_GUI_ENTRIES:
                self.A_matrix = None
                self.b_vector = None
                self.x0_vector = None
        except ValueError:
            self.A_matrix = None
            self.b_vector = None
            self.x0_vector = None
        self._create_matrix_entries()

    def _create_matrix_entries(self):
        try:
            n_val = int(self.n_var.get())
            if not (1 <= n_val):
                messagebox.showerror("Error de Dimensión", "La dimensión N debe ser un entero positivo.")
                self.n_var.set("3");
                n_val = 3
        except ValueError:
            messagebox.showerror("Error de Dimensión", "Dimensión N inválida. Usando N=3 por defecto.")
            self.n_var.set("3");
            n_val = 3

        for widget in self.matrix_entry_frame.winfo_children(): widget.destroy()
        self.matrix_A_entries = [];
        self.vector_b_entries = [];
        self.vector_x0_entries = []

        if n_val > MAX_N_FOR_GUI_ENTRIES:
            ttk.Label(self.matrix_entry_frame,
                      text=f"N={n_val} es muy grande para edición manual.\nCargue los datos desde un archivo.").pack(
                pady=10)
            return

        entries_container = ttk.Frame(self.matrix_entry_frame);
        entries_container.pack(pady=5)
        frame_A = ttk.Frame(entries_container);
        frame_A.pack(side=tk.LEFT, padx=(0, 10), anchor=tk.N)
        ttk.Label(frame_A, text="Matriz A:", font=('Helvetica', 10, 'bold')).pack(pady=(0, 3), anchor=tk.W)
        grid_A = ttk.Frame(frame_A);
        grid_A.pack()
        for i in range(n_val):
            row_entries_A = []
            for j in range(n_val):
                entry = ttk.Entry(grid_A, width=7, font=('Courier New', 10), justify='right')
                entry.grid(row=i, column=j, padx=1, pady=1);
                row_entries_A.append(entry)
            self.matrix_A_entries.append(row_entries_A)

        frame_b = ttk.Frame(entries_container);
        frame_b.pack(side=tk.LEFT, padx=(0, 10), anchor=tk.N)
        ttk.Label(frame_b, text="Vector b:", font=('Helvetica', 10, 'bold')).pack(pady=(0, 3), anchor=tk.W)
        grid_b = ttk.Frame(frame_b);
        grid_b.pack()
        for i in range(n_val):
            entry = ttk.Entry(grid_b, width=7, font=('Courier New', 10), justify='right')
            entry.grid(row=i, column=0, padx=1, pady=1);
            self.vector_b_entries.append(entry)

        frame_x0 = ttk.Frame(entries_container);
        frame_x0.pack(side=tk.LEFT, anchor=tk.N)
        ttk.Label(frame_x0, text="Vector X(0) (iterativos):", font=('Helvetica', 10, 'bold')).pack(pady=(0, 3),
                                                                                                   anchor=tk.W)
        grid_x0 = ttk.Frame(frame_x0);
        grid_x0.pack()
        for i in range(n_val):
            entry = ttk.Entry(grid_x0, width=7, font=('Courier New', 10), justify='right')
            entry.grid(row=i, column=0, padx=1, pady=1);
            entry.insert(0, "0");
            self.vector_x0_entries.append(entry)

        if n_val == 3 and self.A_matrix is None:
            example_A = [[4, 1, -1], [2, 7, 1], [1, -3, 12]];
            example_b = [3, 19, 31]
            for r_idx in range(n_val):
                for c_idx in range(n_val):
                    self.matrix_A_entries[r_idx][c_idx].delete(0, tk.END)
                    self.matrix_A_entries[r_idx][c_idx].insert(0, str(example_A[r_idx][c_idx]))
                if self.vector_b_entries:
                    self.vector_b_entries[r_idx].delete(0, tk.END)
                    self.vector_b_entries[r_idx].insert(0, str(example_b[r_idx]))
                if self.vector_x0_entries:
                    self.vector_x0_entries[r_idx].delete(0, tk.END)
                    self.vector_x0_entries[r_idx].insert(0, "0")

    def get_matrix_from_entries(self, matrix_attr_name="A_matrix", entry_attr_name="matrix_A_entries",
                                matrix_label="A"):
        try:
            n_str = self.n_var.get()
            if not n_str: messagebox.showerror("Error de Dimensión", "Dimensión N no especificada."); return None
            n = int(n_str)
        except ValueError:
            messagebox.showerror("Error de Dimensión", f"Dimensión N '{n_str}' inválida."); return None
        if n <= 0: messagebox.showerror("Error de Dimensión", "N debe ser positivo."); return None

        internal_matrix = getattr(self, matrix_attr_name, None)
        if n > MAX_N_FOR_GUI_ENTRIES:
            if internal_matrix is not None and internal_matrix.shape == (n, n):
                return internal_matrix
            else:
                messagebox.showerror("Error de Datos",
                                     f"Matriz {matrix_label} (N={n}) no cargada y muy grande para GUI."); return None

        entry_widgets = getattr(self, entry_attr_name, [])
        if not entry_widgets or len(entry_widgets) != n or (n > 0 and len(entry_widgets[0]) != n):
            if internal_matrix is not None and internal_matrix.shape == (n, n): return internal_matrix
            messagebox.showerror("Error Interno", f"Entradas para Matriz {matrix_label} (N={n}) no listas.");
            return None

        A = np.zeros((n, n), dtype=float);
        val_str_for_error = ""
        try:
            for i in range(n):
                for j in range(n):
                    val_str = entry_widgets[i][j].get();
                    val_str_for_error = val_str
                    if not val_str.strip(): val_str = "0"
                    try:
                        A[i, j] = eval(val_str, {"__builtins__": None, "math": math, "np": np, "sqrt": math.sqrt,
                                                 "sin": math.sin, "cos": math.cos, "exp": math.exp, "pi": math.pi,
                                                 "e": math.e})
                    except:
                        A[i, j] = float(val_str)
            setattr(self, matrix_attr_name, A);
            return A
        except ValueError:
            messagebox.showerror(f"Error en Matriz {matrix_label}", f"Valor inválido '{val_str_for_error}'."); setattr(
                self, matrix_attr_name, None); return None
        except Exception as e:
            messagebox.showerror(f"Error en Matriz {matrix_label}", f"Error: {e}"); setattr(self, matrix_attr_name,
                                                                                            None); return None

    def get_vector_from_entries(self, vector_attr_name: str, entry_attr_name: str, vector_label: str,
                                fail_if_empty: bool = True) -> Optional[np.ndarray]:
        try:
            n_str = self.n_var.get()
            if not n_str:
                if vector_label == "X(0) (iterativos)" and not fail_if_empty:
                    messagebox.showwarning("Advertencia Dimensión",
                                           "N no especificada para X(0) por defecto. No se puede crear vector nulo.")
                    return None
                messagebox.showerror("Error de Dimensión", "N no especificada.")
                return None
            n = int(n_str)
        except ValueError:
            messagebox.showerror("Error de Dimensión", f"N '{n_str}' inválida.")
            return None
        if n <= 0:
            messagebox.showerror("Error de Dimensión", "N debe ser positivo.")
            return None

        internal_vector = getattr(self, vector_attr_name, None)
        if n > MAX_N_FOR_GUI_ENTRIES:
            if internal_vector is not None and internal_vector.shape == (n,):
                return internal_vector
            else:
                if vector_label == "X(0) (iterativos)" and not fail_if_empty:
                    print(f"Info: Vector X(0) (N={n}, grande) no provisto. Usando nulo por defecto.")
                    default_x0 = np.zeros(n, dtype=float)
                    if vector_attr_name == "x0_vector": self.x0_vector = default_x0  # Almacenar si es para self.x0_vector
                    return default_x0
                if fail_if_empty:
                    messagebox.showerror("Error de Datos",
                                         f"Vector {vector_label} (N={n}) no cargado y muy grande para GUI.")
                return None

        entry_widgets = getattr(self, entry_attr_name, [])
        if not entry_widgets or len(entry_widgets) != n:
            if internal_vector is not None and internal_vector.shape == (n,):
                return internal_vector
            if vector_label == "X(0) (iterativos)" and not fail_if_empty:
                print(f"Info: Entradas GUI de X(0) (N={n}) no listas/coincidentes. Usando nulo por defecto.")
                default_x0 = np.zeros(n, dtype=float)
                if vector_attr_name == "x0_vector": self.x0_vector = default_x0
                return default_x0
            if fail_if_empty:
                messagebox.showerror("Error Interno", f"Entradas para Vector {vector_label} (N={n}) no listas.")
            return None

        v = np.zeros(n, dtype=float);
        all_gui_empty_or_zero = True;
        val_str_for_error = ""
        try:
            for i in range(n):
                val_str = entry_widgets[i].get().strip();
                val_str_for_error = val_str
                current_val_num = 0.0
                if not val_str:  # Campo vacío
                    if fail_if_empty and vector_label != "X(0) (iterativos)":
                        messagebox.showerror(f"Error en Vector {vector_label}",
                                             f"Entrada vacía para {vector_label}[{i + 1}].");
                        setattr(self, vector_attr_name, None);
                        return None
                    # Si es X0 opcional o no falla si está vacío, se asume 0.0
                else:  # Campo no vacío
                    try:
                        current_val_num = eval(val_str,
                                               {"__builtins__": None, "math": math, "np": np, "sqrt": math.sqrt,
                                                "sin": math.sin, "cos": math.cos, "exp": math.exp, "pi": math.pi,
                                                "e": math.e})
                    except:
                        current_val_num = float(val_str)  # Puede lanzar ValueError
                    if not np.isclose(current_val_num, 0.0, atol=ZERO_TOLERANCE):
                        all_gui_empty_or_zero = False
                v[i] = current_val_num

            final_vector = v
            if vector_label == "X(0) (iterativos)" and all_gui_empty_or_zero and not fail_if_empty:
                print(f"Info: X(0) (N={n}) ingresado como vacío/cero en GUI. Usando nulo por defecto.")
                final_vector = np.zeros(n, dtype=float)

            setattr(self, vector_attr_name, final_vector);
            return final_vector
        except ValueError:
            if vector_label == "X(0) (iterativos)" and not fail_if_empty:
                print(f"Advertencia: Valor inválido '{val_str_for_error}' para X(0). Usando nulo por defecto.")
                final_vector = np.zeros(n, dtype=float);
                setattr(self, vector_attr_name, final_vector);
                return final_vector
            if fail_if_empty: messagebox.showerror(f"Error en Vector {vector_label}",
                                                   f"Valor inválido '{val_str_for_error}'.");
            setattr(self, vector_attr_name, None);
            return None
        except Exception as e:
            if vector_label == "X(0) (iterativos)" and not fail_if_empty:
                print(f"Advertencia: Error '{e}' al leer X(0). Usando nulo por defecto.")
                final_vector = np.zeros(n, dtype=float);
                setattr(self, vector_attr_name, final_vector);
                return final_vector
            if fail_if_empty: messagebox.showerror(f"Error en Vector {vector_label}", f"Error: {e}");
            setattr(self, vector_attr_name, None);
            return None

    def load_data_from_file(self, data_type: str):
        filepath = filedialog.askopenfilename(
            title=f"Cargar {data_type} desde archivo",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("NumPy files", "*.npy"), ("All files", "*.*")])
        if not filepath: return
        try:
            if filepath.endswith('.npy'):
                data = np.load(filepath)
            else:
                try:
                    data = np.loadtxt(filepath, delimiter=',', ndmin=1 if data_type != 'A' else 2)
                except ValueError:
                    try:
                        data = np.loadtxt(filepath, delimiter=' ', ndmin=1 if data_type != 'A' else 2)
                    except ValueError:
                        try:
                            data = np.loadtxt(filepath, ndmin=1 if data_type != 'A' else 2)
                        except ValueError as e_load:
                            messagebox.showerror("Error de Carga",
                                                 f"No se pudo interpretar '{filepath.split('/')[-1]}'.\nError: {e_load}"); return

            current_n_str = self.n_var.get();
            current_n = None
            if current_n_str:
                try:
                    current_n = int(current_n_str)
                except ValueError:
                    pass

            if data_type == 'A':
                if data.ndim != 2 or data.shape[0] != data.shape[1]: messagebox.showerror("Error de Formato",
                                                                                          "Archivo para A no es matriz cuadrada."); return
                N_loaded = data.shape[0]
                self.n_var.set(str(N_loaded));
                self.A_matrix = data.astype(float)
                if current_n is not None and N_loaded != current_n:
                    self.b_vector = None;
                    self.x0_vector = None
                    if hasattr(self, 'vector_b_entries'): self.vector_b_entries = []
                    if hasattr(self, 'vector_x0_entries'): self.vector_x0_entries = []
                self.create_matrix_entries_event_wrapper()
                if N_loaded <= MAX_N_FOR_GUI_ENTRIES:
                    for r in range(N_loaded):
                        for c in range(N_loaded):
                            if r < len(self.matrix_A_entries) and c < len(self.matrix_A_entries[r]):
                                self.matrix_A_entries[r][c].delete(0, tk.END);
                                self.matrix_A_entries[r][c].insert(0, str(self.A_matrix[r, c]))
                self.display_output("Carga de Archivo",
                                    {"status": f"Matriz A (N={N_loaded}) cargada: {filepath.split('/')[-1]}"})
            elif data_type in ['b', 'X0']:
                if not current_n: messagebox.showerror("Error de Dimensión",
                                                       "Defina N (cargando A) antes de cargar vector."); return
                if data.ndim == 2:
                    if data.shape[0] == 1 and data.shape[1] == current_n:
                        data = data.flatten()
                    elif data.shape[1] == 1 and data.shape[0] == current_n:
                        data = data.flatten()
                    else:
                        messagebox.showerror("Error de Formato",
                                             f"Archivo para {data_type} no es 1x{current_n} o {current_n}x1."); return
                elif data.ndim == 1 and data.shape[0] != current_n:
                    messagebox.showerror("Error de Formato",
                                         f"Archivo para {data_type} no tiene longitud N={current_n}."); return
                elif data.ndim != 1:
                    messagebox.showerror("Error de Formato", f"Archivo para {data_type} no es vector."); return

                vec_data = data.astype(float)
                if data_type == 'b':
                    self.b_vector = vec_data
                elif data_type == 'X0':
                    self.x0_vector = vec_data
                if current_n <= MAX_N_FOR_GUI_ENTRIES:
                    target_entries = self.vector_b_entries if data_type == 'b' else self.vector_x0_entries
                    if target_entries and len(target_entries) == current_n:
                        for i in range(current_n): target_entries[i].delete(0, tk.END); target_entries[i].insert(0,
                                                                                                                 str(
                                                                                                                     vec_data[
                                                                                                                         i]))
                self.display_output("Carga de Archivo", {
                    "status": f"Vector {data_type} (N={current_n}) cargado: {filepath.split('/')[-1]}"})
        except Exception as e:
            messagebox.showerror("Error al Cargar",
                                 f"No se pudo cargar '{filepath.split('/')[-1]}':\n{type(e).__name__}: {e}")
            self.display_output("Error al Cargar", {
                "status": f"No se pudo cargar {filepath.split('/')[-1]}: {e}\n{traceback.format_exc()}",
                "tag": "error"})

    def save_matrix_to_file(self, matrix_type: str = 'A'):
        if matrix_type == 'A':
            matrix_to_save = self.get_matrix_from_entries(matrix_attr_name="A_matrix",
                                                          entry_attr_name="matrix_A_entries", matrix_label="A")
            if matrix_to_save is None: messagebox.showerror("Error al Guardar", "No hay Matriz A válida."); return
            default_name = f"matriz_A_{matrix_to_save.shape[0]}x{matrix_to_save.shape[0]}.txt"
        else:
            messagebox.showinfo("Info", "Guardado no implementado para este tipo."); return
        filepath = filedialog.asksaveasfilename(title=f"Guardar {matrix_type} como...", defaultextension=".txt",
                                                initialfile=default_name,
                                                filetypes=[("Text files (espacio)", "*.txt"), ("NumPy binary", "*.npy"),
                                                           ("CSV (coma)", "*.csv")])
        if not filepath: return
        try:
            prec = self.current_display_precision
            if filepath.endswith('.npy'):
                np.save(filepath, matrix_to_save)
            elif filepath.endswith('.csv'):
                np.savetxt(filepath, matrix_to_save, fmt=f'%.{prec}e', delimiter=',')
            else:
                np.savetxt(filepath, matrix_to_save, fmt=f'%.{prec}e', delimiter=' ')
            self.display_output("Guardar Archivo",
                                {"status": f"Datos '{matrix_type}' guardados en: {filepath.split('/')[-1]}",
                                 "tag": "success"})
        except Exception as e:
            messagebox.showerror("Error al Guardar", f"No se pudo guardar '{filepath.split('/')[-1]}':\n{e}")
            self.display_output("Error al Guardar",
                                {"status": f"No se pudo guardar: {e}\n{traceback.format_exc()}", "tag": "error"})

    def display_output(self, title: str, content_data: Dict[str, Any], tag: str = 'info'):
        self.output_text.configure(state='normal')
        if float(self.output_text.index(tk.END)) > 70.0: self.output_text.see(tk.END)
        status_tag = tag
        if "status" in content_data:  # Asumir que status siempre está
            status_lower = str(content_data["status"]).lower()
            if "error" in status_lower:
                status_tag = 'error'
            elif "advertencia" in status_lower:
                status_tag = 'warning'
            elif "exitosa" in status_lower or "completada" in status_lower or "calculado" in status_lower or "encontrada" in status_lower:
                status_tag = 'success'

        self.output_text.insert(tk.END, f"--- {title} ---\n", ('title', status_tag))
        prec = self.current_display_precision

        if "history" in content_data and isinstance(content_data["history"], list):
            self.last_iteration_history = content_data["history"]
            if self.last_iteration_history:
                self.export_iter_button.config(state=tk.NORMAL)
                self.last_iter_params['TOL'] = content_data.get('TOL_used', self.last_iter_params.get('TOL', 'N/A'))
                self.last_iter_params['NMAX'] = content_data.get('NMAX_used', self.last_iter_params.get('NMAX', 'N/A'))
                if 'omega_used' in content_data:
                    self.last_iter_params['omega'] = content_data['omega_used']
                else:
                    self.last_iter_params.pop('omega', None)
            else:
                self.export_iter_button.config(state=tk.DISABLED)
        elif not ("history" in content_data and isinstance(content_data["history"], list)) and \
                title.lower() not in ["exportación", "carga de archivo", "guardar archivo", "info", "análisis",
                                      "aplicar precisión"]:
            self.last_iteration_history = [];
            self.export_iter_button.config(state=tk.DISABLED)

        preferred_order = ["status", "X_solution", "determinant_value", "inverse_matrix", "condition_number",
                           "norm_used_for_cond", "norms", "eigenvalues_result", "eigenvectors_result",
                           "Q_matrix_result", "R_matrix_result", "U_svd_matrix_result", "S_svd_matrix_result",
                           "S_svd_values_vector", "Vh_svd_matrix_result", "iterations_taken", "final_error", "info",
                           "details_text", "L_final", "U_final", "P_factor", "L_chol_matrix",
                           "final_A_upper_triangular", "final_b_transformed", "final_NROW (0-based)",
                           "final_augmented_matrix [I|X]"]

        for key in preferred_order:
            if key in content_data:
                value = content_data[key]
                key_display_name = key.replace("_value", "").replace("_result", "").replace("_matrix", "").replace(
                    "_vector", "").replace("_values", "").replace("X_solution", "Solución X").replace("_",
                                                                                                      " ").capitalize()
                current_tag_to_use = status_tag if key == "status" else tag  # Usar status_tag para el status, default para otros

                if key == "norms" and isinstance(value, dict):
                    self.output_text.insert(tk.END, f"{key_display_name}:\n", ('matrix_label',))
                    for norm_key, norm_value in value.items():
                        norm_key_display = norm_key.replace("_", " ").replace("A norm", "||A||").replace("b norm",
                                                                                                         "||b||").replace(
                            "euclidiana", "2")
                        if isinstance(norm_value, str):
                            self.output_text.insert(tk.END, f"  {norm_key_display}: {norm_value}\n")
                        else:
                            self.output_text.insert(tk.END, f"  {norm_key_display}: {norm_value:.{prec}e}\n")
                elif isinstance(value, np.ndarray):
                    self.output_text.insert(tk.END, f"{key_display_name}:\n", ('matrix_label',));
                    self.output_text.insert(tk.END,
                                            f"{np.array2string(value, precision=prec, suppress_small=True, max_line_width=np.get_printoptions()['linewidth'])}\n")
                elif isinstance(value, (float, np.floating, int, np.integer)):
                    self.output_text.insert(tk.END, f"{key_display_name}: {value:.{prec}e}\n")
                elif isinstance(value, list) and key == "final_NROW (0-based)":  # Caso especial para NROW
                    self.output_text.insert(tk.END, f"{key_display_name}:\n{value}\n", ('matrix_label',))
                elif value is not None:  # Para status, info, details_text, etc. y otros valores
                    self.output_text.insert(tk.END, f"{key_display_name}: {value}\n", (current_tag_to_use,))
        self.output_text.insert(tk.END, "\n")

        if self.show_iter_history_in_gui_var.get() and "history" in content_data and isinstance(content_data["history"],
                                                                                                list) and content_data[
            "history"]:
            self.output_text.insert(tk.END, "Historial Iteraciones (k, X^(k), ||Error||_inf):\n", ('matrix_label',));
            history_list = content_data["history"]
            try:
                display_limit = int(self.iter_display_limit_spinbox.get())
            except ValueError:
                display_limit = 10
            if display_limit <= 0: display_limit = len(history_list)
            # ... (lógica de visualización de historial como la tenías) ...
            if len(history_list) > display_limit and display_limit > 5:
                num_start_end = max(2, display_limit // 3)
                for iter_data_tuple in history_list[:num_start_end]: self.output_text.insert(tk.END,
                                                                                             f"k={iter_data_tuple[0]:3d}, X={np.array2string(iter_data_tuple[1], precision=prec, suppress_small=True)}, err={iter_data_tuple[2]:.{prec}e}\n")
                self.output_text.insert(tk.END,
                                        f"...\n({len(history_list) - 2 * num_start_end} iteraciones omitidas)\n...\n")
                for iter_data_tuple in history_list[-num_start_end:]: self.output_text.insert(tk.END,
                                                                                              f"k={iter_data_tuple[0]:3d}, X={np.array2string(iter_data_tuple[1], precision=prec, suppress_small=True)}, err={iter_data_tuple[2]:.{prec}e}\n")
            else:
                for iter_data_tuple in history_list[:display_limit]: self.output_text.insert(tk.END,
                                                                                             f"k={iter_data_tuple[0]:3d}, X={np.array2string(iter_data_tuple[1], precision=prec, suppress_small=True)}, err={iter_data_tuple[2]:.{prec}e}\n")
            self.output_text.insert(tk.END, "\n")

        if self.show_detailed_steps_var.get() and "steps_detailed" in content_data and isinstance(
                content_data["steps_detailed"], list):
            self.output_text.insert(tk.END, "\n--- Pasos Detalladísimos ---\n")
            for step_detail in content_data["steps_detailed"]: self.output_text.insert(tk.END, f"{step_detail}\n")

        self.output_text.insert(tk.END, "\n" + "=" * 80 + "\n\n");
        self.output_text.configure(state='disabled');
        self.output_text.see(tk.END)

    def clear_output(self):
        self.output_text.configure(state='normal');
        self.output_text.delete('1.0', tk.END);
        self.output_text.configure(state='disabled')
        self.last_iteration_history = [];
        self.last_iter_params = {};
        self.export_iter_button.config(state=tk.DISABLED)

    def apply_output_settings(self):
        try:
            prec_str = self.decimal_precision_var.get()
            if not prec_str.strip(): messagebox.showwarning("Precisión Inválida", "Campo vacío."); return
            prec = int(prec_str)
            if not (0 <= prec <= 16): messagebox.showwarning("Precisión Inválida",
                                                             "Decimales deben estar entre 0-16."); self.decimal_precision_var.set(
                str(self.current_display_precision)); return
            current_linewidth = np.get_printoptions().get('linewidth', 160)
            np.set_printoptions(precision=prec, suppress=True, linewidth=current_linewidth)
            self.current_display_precision = prec
            # No mostrar en GUI principal, solo en consola para evitar bucles si display_output se modifica mucho.
            # print(f"Info: Precisión de salida NumPy actualizada a {prec} decimales.")
        except ValueError:
            messagebox.showerror("Error de Precisión", "Decimales inválidos.");
            self.decimal_precision_var.set(str(self.current_display_precision))
            # Re-aplicar la precisión válida actual en caso de error
            np.set_printoptions(precision=self.current_display_precision, suppress=True,
                                linewidth=np.get_printoptions().get('linewidth', 160))

    def export_iteration_history(self):
        if not self.last_iteration_history: messagebox.showinfo("Info", "No hay historial para exportar."); return

        # Validar la estructura del historial antes de proceder
        if not all(isinstance(item, tuple) and len(item) == 3 and isinstance(item[1], np.ndarray) for item in
                   self.last_iteration_history):
            messagebox.showerror("Error de Datos", "El formato del historial de iteraciones es incorrecto.")
            return

        filepath = filedialog.asksaveasfilename(title="Exportar Historial de Iteraciones", defaultextension=".txt",
                                                initialfile="historial_iteraciones.txt",
                                                filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"),
                                                           ("All files", "*.*")])
        if not filepath: return
        try:
            prec = self.current_display_precision;
            N_cols = 0
            # Asegurar que el primer elemento del historial (si existe) tiene el formato esperado para N_cols
            if self.last_iteration_history and self.last_iteration_history[0][1].ndim == 1:
                N_cols = len(self.last_iteration_history[0][1])

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Historial Iteraciones:\n");
                f.write(f"TOL: {self.last_iter_params.get('TOL', 'N/A')}\n");
                f.write(f"NMAX: {self.last_iter_params.get('NMAX', 'N/A')}\n")
                if 'omega' in self.last_iter_params: f.write(f"Omega (ω): {self.last_iter_params['omega']:.{prec}f}\n")
                f.write(f"Precisión: {prec}\n{'=' * 100}\n");
                is_csv = filepath.endswith('.csv');
                delimiter = ',' if is_csv else '\t'
                header_k = "k";
                header_X_cols = [f"X_{i + 1}" for i in range(N_cols)];
                header_err = "Error_inf"
                f.write(header_k + delimiter + delimiter.join(header_X_cols) + delimiter + header_err + "\n")
                for k_iter, x_val_arr, err_val in self.last_iteration_history:
                    if isinstance(x_val_arr, np.ndarray) and x_val_arr.ndim == 1 and len(x_val_arr) == N_cols:
                        x_str_elements = [f"{val_i:.{prec}e}" for val_i in x_val_arr]
                        f.write(f"{k_iter}{delimiter}{delimiter.join(x_str_elements)}{delimiter}{err_val:.{prec}e}\n")
                    else:
                        f.write(
                            f"{k_iter}{delimiter}ERROR_FORMATO_VECTOR_X{delimiter}{err_val:.{prec}e}\n")  # Log de error
            self.display_output("Exportación",
                                {"status": f"Historial guardado en:\n{filepath.split('/')[-1]}", "tag": "success"})
        except Exception as e:
            messagebox.showerror("Error al Exportar", f"No se pudo guardar: {e}");
            self.display_output("Error al Exportar",
                                {"status": f"No se pudo guardar: {e}\n{traceback.format_exc()}", "tag": "error"})

    def export_full_output_to_txt(self):
        content_to_save = self.output_text.get("1.0", tk.END).strip()
        if not content_to_save: messagebox.showinfo("Exportar Resultados", "No hay resultados para exportar."); return
        filepath = filedialog.asksaveasfilename(title="Guardar Resultados Completos Como...", defaultextension=".txt",
                                                initialfile="resultados_completos.txt",
                                                filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not filepath: return
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(
                    f"Resultados exportados el: {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 80}\n\n{content_to_save}")
            messagebox.showinfo("Exportación Exitosa", f"Resultados guardados en:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error al Exportar", f"No se pudo guardar el archivo:\n{e}")

    def execute_method_analysis_simple(self, analysis_name: str, analysis_func: Callable):
        if self.is_calculating: messagebox.showwarning("Ocupado", "Cálculo en progreso."); return
        self.clear_output();
        self.apply_output_settings()
        A = self.get_matrix_from_entries(matrix_attr_name="A_matrix", entry_attr_name="matrix_A_entries",
                                         matrix_label="A")
        if A is None: return

        # Estos análisis son rápidos, no necesitan hilo por ahora.
        start_time = time.time()
        try:
            result_val = analysis_func(A)  # Para es_simetrica, es_definida_positiva
            details_to_show: Dict[str, Any] = {}
            tag = 'info'

            if analysis_name == "Dominancia Diagonal":  # Esta función devuelve (summary_str, details_text_str)
                summary_text, details_text_str = result_val  # Desempaquetar
                details_to_show["status"] = summary_text
                details_to_show["details_text"] = details_text_str
                if "ESTRICTAMENTE" in summary_text or "DÉBILMENTE" in summary_text and "NO" not in summary_text:
                    tag = 'success'
                elif "NO" in summary_text:
                    tag = 'warning'  # O 'info'
            else:  # Para simetría y definida positiva
                status_str = f"La matriz A {'ES' if result_val else 'NO ES'} {analysis_name.lower().replace('verificar ', '')}."
                details_to_show["status"] = status_str
                tag = 'success' if result_val else 'info'

            details_to_show["tag"] = tag
            self.display_output(f"Análisis: {analysis_name}", details_to_show)

        except Exception as e:
            messagebox.showerror("Error en Análisis",
                                 f"Ocurrió un error en '{analysis_name}':\n{type(e).__name__}: {e}")
            tb_str = traceback.format_exc()
            self.display_output(f"Excepción en {analysis_name}",
                                {"status": f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{tb_str}", "tag": "error"})
            print(f"EXCEPCIÓN en {analysis_name}:\n{tb_str}")

        end_time = time.time()
        self.output_text.configure(state='normal')
        self.output_text.insert(tk.END, f"--- Tiempo de análisis: {end_time - start_time:.6f} segundos. ---\n",
                                ('info',))
        self.output_text.configure(state='disabled');
        self.output_text.see(tk.END)

    def execute_matrix_operation(self, op_name: str, op_func: Callable, requires_A: bool = True,
                                 requires_b_optional: bool = False, pass_norm_type: bool = False):
        if self.is_calculating: messagebox.showwarning("Ocupado", "Cálculo en progreso."); return
        self.clear_output();
        self.apply_output_settings()

        A: Optional[np.ndarray] = None
        b: Optional[np.ndarray] = None
        if requires_A:
            A = self.get_matrix_from_entries(matrix_attr_name="A_matrix", entry_attr_name="matrix_A_entries",
                                             matrix_label="A")
            if A is None: return
        if requires_b_optional:
            b = self.get_vector_from_entries(vector_attr_name="b_vector", entry_attr_name="vector_b_entries",
                                             vector_label="b", fail_if_empty=False)
            if b is None and op_name == "Normas": messagebox.showwarning("Advertencia Vector b",
                                                                         "Vector b no válido/provisto. Solo normas de A.")

        args_for_worker: Tuple = ()
        kwargs_for_worker: Dict[str, Any] = {}
        if op_name == "Normas":
            kwargs_for_worker = {"A_matrix": A, "b_vector": b}
        elif pass_norm_type:
            norm_type_str = self.cond_norm_var.get();
            p_norm_val: Union[int, float, str] = 2
            if norm_type_str == "inf":
                p_norm_val = np.inf
            elif norm_type_str == "-inf":
                p_norm_val = -np.inf
            elif norm_type_str == "fro":
                p_norm_val = 'fro'
            elif norm_type_str in ["1", "2", "-1", "-2"]:
                p_norm_val = int(norm_type_str)
            kwargs_for_worker = {"A_matrix": A, "p_norm": p_norm_val}
        else:  # Det, Inv, Eig, QR, SVD
            if A is not None: args_for_worker = (A,)

        self.start_calculation_feedback(f"Calculando {op_name}...")
        self.calculation_thread = threading.Thread(target=self.calculation_worker,
                                                   args=(op_name, op_func, args_for_worker, kwargs_for_worker),
                                                   daemon=True)
        self.calculation_thread.start()

    def execute_method(self, method_name: str, method_func: Optional[Callable],
                       # method_func puede ser None para Cholesky/LU Solve
                       requires_b: bool = True, requires_X0: bool = False, requires_omega: bool = False,
                       is_factorization: bool = False, is_analysis: bool = False):
        if self.is_calculating: messagebox.showwarning("Ocupado", "Cálculo en progreso."); return
        self.clear_output();
        self.apply_output_settings()

        A: Optional[np.ndarray] = None
        if not (is_analysis and method_name not in ["Dominancia Diagonal"]):
            A = self.get_matrix_from_entries(matrix_attr_name="A_matrix", entry_attr_name="matrix_A_entries",
                                             matrix_label="A")
            if A is None: return

        b: Optional[np.ndarray] = None
        if requires_b:
            b = self.get_vector_from_entries(vector_attr_name="b_vector", entry_attr_name="vector_b_entries",
                                             vector_label="b", fail_if_empty=True)
            if b is None: return

        X0: Optional[np.ndarray] = None;
        tol: Optional[float] = None;
        nmax: Optional[int] = None;
        omega_val: Optional[float] = None
        if requires_X0:
            X0 = self.get_vector_from_entries(vector_attr_name="x0_vector", entry_attr_name="vector_x0_entries",
                                              vector_label="X(0) (iterativos)", fail_if_empty=False)
            if X0 is None: messagebox.showerror("Error Parámetros",
                                                "No se pudo obtener/crear X(0) para métodos iterativos."); return
            try:
                tol_str = self.tol_entry.get();
                nmax_str = self.n_iter_entry.get()
                if not tol_str or not nmax_str: raise ValueError("TOL o NMAX vacíos.")
                tol = float(tol_str);
                nmax = int(nmax_str)
                if tol <= 0 or nmax <= 0: messagebox.showerror("Error Parámetros",
                                                               "TOL y NMAX deben ser positivos."); return
            except ValueError:
                messagebox.showerror("Error Parámetros", "TOL o NMAX inválidos."); return
        if requires_omega:
            try:
                omega_str = self.omega_entry.get()
                if not omega_str: raise ValueError("Omega vacío.")
                omega_val = float(omega_str)
                if not (0 < omega_val < 2): messagebox.showwarning("Advertencia Parámetro SOR",
                                                                   "Omega (ω) para SOR usualmente está en (0, 2).");
            except ValueError:
                messagebox.showerror("Error Parámetros", "Omega (ω) inválido."); return

        args_for_worker: Tuple = ()
        kwargs_for_worker: Dict[str, Any] = {}
        target_func_for_worker: Optional[Callable] = method_func

        # Preparar argumentos y función para el worker
        if is_factorization:
            if A is not None:
                args_for_worker = (A,)
            else:
                messagebox.showerror("Error", "Matriz A necesaria para factorización."); return
        elif is_analysis:  # Para Dominancia Diagonal
            if A is not None:
                args_for_worker = (A,)
            else:
                messagebox.showerror("Error", "Matriz A necesaria para análisis."); return
        else:  # Métodos de solución
            current_b = np.copy(b) if b is not None else None
            if current_b is None and requires_b: messagebox.showerror("Error",
                                                                      f"Vector b necesario para {method_name}."); return

            if method_name == "Eliminación Gaussiana":
                if A is None or current_b is None: return  # Chequeo adicional
                kwargs_for_worker = {"A_matrix_orig": A, "b_vector_orig": current_b,
                                     "use_scaled_partial_pivoting": self.use_scaled_pivoting_var.get()}
            elif method_name == "Gauss-Jordan":
                if A is None or current_b is None: return
                kwargs_for_worker = {"A_matrix_orig": A, "b_vector_orig": current_b}
            elif requires_X0 and requires_omega:  # SOR
                if A is None or current_b is None or X0 is None or omega_val is None or tol is None or nmax is None: return
                kwargs_for_worker = {"A_matrix": A, "b_vector": current_b, "X0_vector": X0, "omega": omega_val,
                                     "TOL": tol, "NMAX": nmax}
            elif requires_X0:  # Jacobi, Gauss-Seidel
                if A is None or current_b is None or X0 is None or tol is None or nmax is None: return
                kwargs_for_worker = {"A_matrix": A, "b_vector": current_b, "X0_vector": X0, "TOL": tol, "NMAX": nmax}
            elif method_name == "LU Solve":  # Renombrado para claridad
                if self.L_matrix_factor is None or self.U_matrix_factor is None or current_b is None: messagebox.showerror(
                    "Error Previo", "Realice Factorización LU y asegure vector b."); return
                target_func_for_worker = _solve_lu_burden_worker  # Usar el worker wrapper
                kwargs_for_worker = {"L_matrix": self.L_matrix_factor, "U_matrix": self.U_matrix_factor,
                                     "P_factor": self.P_factor, "b_vector_orig": current_b}
            elif method_name == "Cholesky Solve":
                if self.L_chol_matrix is None or current_b is None: messagebox.showerror("Error Previo",
                                                                                         "Realice Factorización Cholesky y asegure vector b."); return
                target_func_for_worker = _cholesky_solve_burden_worker  # Usar el worker wrapper
                kwargs_for_worker = {"L_chol_matrix": self.L_chol_matrix, "b_vector_orig": current_b}

        if target_func_for_worker is None:  # Debería haberse manejado antes si method_func era None
            messagebox.showerror("Error Interno", f"Función de cálculo no definida para {method_name}");
            return

        self.start_calculation_feedback(f"Calculando {method_name}...")
        self.calculation_thread = threading.Thread(target=self.calculation_worker,
                                                   args=(method_name, target_func_for_worker, args_for_worker,
                                                         kwargs_for_worker), daemon=True)
        self.calculation_thread.start()


# ===========================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ===========================================================================
if __name__ == '__main__':
    root = tk.Tk()
    try:
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(family="Helvetica", size=10)
        root.option_add("*Font", default_font)
        actual_family = default_font.actual("family")
        actual_size = default_font.actual("size")
        font_string_for_text = f"{{{actual_family}}} {actual_size}"
        root.option_add("*Text*Font", font_string_for_text)
        root.option_add("*ScrolledText*Font", font_string_for_text)  # Específico para ScrolledText
    except tk.TclError as e:
        print(f"Advertencia: No se pudo configurar la fuente predeterminada globalmente: {e}. Usando defaults de Tk.")
    except Exception as e_font:
        print(f"Error al configurar la fuente: {e_font}. Usando defaults de Tk.")

    # Definiciones de constantes globales que podrían haberse perdido
    DEFAULT_PRINT_PRECISION = 6
    ZERO_TOLERANCE = 1e-15
    SMALL_PIVOT_WARNING_TOLERANCE = 1e-9
    MAX_N_FOR_GUI_ENTRIES = 20

    # Aquí deberían estar tus funciones numéricas globales (es_simetrica_burden, etc.)
    # ...

    app = NumericalAnalysisApp(root)
    root.mainloop()

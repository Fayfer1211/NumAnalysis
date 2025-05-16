import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QGridLayout, QGroupBox, QMessageBox, QFrame,
    QScrollArea 
)
from PyQt6.QtGui import QFont, QPalette, QColor
from PyQt6.QtCore import Qt
import time 

# --- Importar los métodos del otro archivo ---
try:
    # Asumimos que las funciones de validación como es_simetrica están aquí
    import solution_methods_linear_systems_equations as methods 
except ImportError:
    print("Error: No se pudo encontrar el archivo 'solution_methods_linear_systems_equations.py'.")
    print("Asegúrate de que esté en el mismo directorio que este script.")
    print("Se usarán implementaciones básicas de validación si están disponibles.")
    # Definir funciones básicas si el módulo no se encuentra, para que el resto funcione
    class methods:
        @staticmethod
        def es_simetrica(A, tol=1e-8):
            if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1]: return False
            return np.allclose(A, A.T, atol=tol)
        # Podrías añadir otras funciones necesarias aquí si fallara la importación

# --- Implementación del Descenso Pronunciado ---
# Lo incluimos aquí directamente ya que no estaba en el archivo importado
def steepest_descent(A, b, x0=None, tol=1e-10, max_iter=1000):
    """Método del Descenso Pronunciado (Steepest Descent). Cuenta operaciones."""
    n = len(b)
    if A.shape[0] != n or A.shape[1] != n:
        raise ValueError("La matriz A debe ser cuadrada (n x n) para este método.")
    # Es crucial para la convergencia teórica que A sea SPD
    if not methods.es_simetrica(A):
         print("Advertencia: La matriz A no es simétrica. El Descenso Pronunciado podría no converger o hacerlo lentamente.")

    A = A.astype(float)
    b = b.astype(float)
    x = np.zeros_like(b, dtype=float) if x0 is None else np.array(x0, dtype=float)
    ops_total = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0, 'norm': 0, 'matmul': 0, 'dot': 0}

    # --- Costo Inicial ---
    Ax = A @ x
    ops_total['matmul'] += 1; ops_total['mul'] += n * n; ops_total['add_sub'] += n * (n - 1) if n > 1 else 0 
    r = b - Ax
    ops_total['add_sub'] += n 

    # --- Bucle Principal de Iteraciones ---
    for k in range(max_iter):
        norm_r = np.linalg.norm(r)
        ops_total['norm'] += 1; ops_total['mul'] += n; ops_total['add_sub'] += (n-1) if n > 1 else 0 
        ops_total['compare'] += 1 
        if norm_r < tol:
            return x, k, True, ops_total 

        Ar = A @ r
        ops_total['matmul'] += 1; ops_total['mul'] += n * n; ops_total['add_sub'] += n * (n - 1) if n > 1 else 0

        rTr = r @ r 
        ops_total['dot'] += 1; ops_total['mul'] += n; ops_total['add_sub'] += (n-1) if n > 1 else 0 

        rTAr = r @ Ar 
        ops_total['dot'] += 1; ops_total['mul'] += n; ops_total['add_sub'] += (n-1) if n > 1 else 0

        if abs(rTAr) < 1e-15: 
             print(f"\nSteepest Descent Error: Denominador rTAr = {rTAr:.2e} cercano a cero en iteración {k+1}.")
             if rTAr <= 0 and methods.es_simetrica(A): 
                 print("  Esto puede indicar que la matriz no es definida positiva.")
             return x, k + 1, False, ops_total 
        ops_total['abs'] += 1; ops_total['compare'] += 1
        alpha = rTr / rTAr
        ops_total['div'] += 1

        x = x + alpha * r
        ops_total['mul'] += n; ops_total['add_sub'] += n 

        r = r - alpha * Ar 
        ops_total['mul'] += n; ops_total['add_sub'] += n 

    norm_r_final = np.linalg.norm(r) 
    ops_total['norm'] += 1; ops_total['mul'] += n; ops_total['add_sub'] += (n-1) if n > 1 else 0 
    print(f"Steepest Descent no convergió en {max_iter} iteraciones. Norma residuo final: {norm_r_final:.4e}")
    return x, max_iter, False, ops_total


class AlgebraLinealPyQtGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.A = None
        self.b = None
        self.L = None 
        self.U = None 
        self.L_chol = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Calculadora de Álgebra Lineal Profesional (PyQt6 + Métodos)")
        self.resize(1100, 900) 

        # --- Paleta de Colores y Fuentes ---
        self.colors = {
            "bg_main": "#f0f2f5", "bg_groupbox": "#ffffff", "bg_entry": "#ffffff",
            "text_main": "#2c3e50", "text_label": "#34495e", "button_bg": "#3498db",
            "button_fg": "#ffffff", "button_hover_bg": "#2980b9", "clear_button_bg": "#e74c3c",
            "clear_button_hover_bg": "#c0392b", "result_bg": "#ecf0f1", "result_fg": "#2c3e50",
            "border": "#bdc3c7", "groupbox_title_fg": "#2980b9"
        }
        self.fonts = {
            "main": QFont("Helvetica", 11), "label": QFont("Helvetica", 10, QFont.Weight.Bold),
            "entry": QFont("Helvetica", 10), "button": QFont("Helvetica", 9, QFont.Weight.Bold), 
            "result": QFont("Consolas", 11), "group_title": QFont("Helvetica", 13, QFont.Weight.Bold),
        }
        self.setFont(self.fonts["main"])

        # --- Contenedor Principal para Scroll ---
        scroll_area = QScrollArea(self) 
        scroll_area.setWidgetResizable(True) 
        scroll_area.setStyleSheet(f"QScrollArea {{ border: none; background-color: {self.colors['bg_main']}; }}") 

        scroll_content_widget = QWidget()
        scroll_content_widget.setStyleSheet(f"QWidget {{ background-color: {self.colors['bg_main']}; }}") 

        main_layout = QVBoxLayout(scroll_content_widget) 
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # --- Contenedor para Entradas ---
        input_frame = QFrame() 
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(0,0,0,0)
        input_layout.setSpacing(20)

        # --- Columna Izquierda ---
        left_column_layout = QVBoxLayout()
        left_column_layout.setSpacing(15)

        # --- Sección de Vectores ---
        vector_groupbox = QGroupBox("Vectores / Vector b (Ax=b)")
        vector_groupbox.setFont(self.fonts["group_title"])
        vector_groupbox.setStyleSheet(self._get_groupbox_style())
        vector_form_layout = QGridLayout() 
        vector_form_layout.setSpacing(10) 

        # Crear y configurar Labels explícitamente
        self.vec_a_label = QLabel("Vector A (ej: 1,2,3):")
        self.vec_a_label.setFont(self.fonts["label"])
        self.vec_a_label.setStyleSheet(f"color: {self.colors['text_label']};") # Color explícito

        self.vec_a_entry = QLineEdit()
        self.vec_a_entry.setFont(self.fonts["entry"])
        self.vec_a_entry.setStyleSheet(self._get_lineedit_style())

        self.vec_b_label = QLabel("Vector B (o b para Ax=b):")
        self.vec_b_label.setFont(self.fonts["label"])
        self.vec_b_label.setStyleSheet(f"color: {self.colors['text_label']};") # Color explícito

        self.vec_b_entry = QLineEdit()
        self.vec_b_entry.setFont(self.fonts["entry"])
        self.vec_b_entry.setStyleSheet(self._get_lineedit_style())
        
        self.scalar_v_label = QLabel("Escalar (Vectores):")
        self.scalar_v_label.setFont(self.fonts["label"])
        self.scalar_v_label.setStyleSheet(f"color: {self.colors['text_label']};") # Color explícito

        self.scalar_v_entry = QLineEdit()
        self.scalar_v_entry.setFont(self.fonts["entry"])
        self.scalar_v_entry.setStyleSheet(self._get_lineedit_style())
        self.scalar_v_entry.setFixedWidth(180) 

        vector_form_layout.addWidget(self.vec_a_label, 0, 0)
        vector_form_layout.addWidget(self.vec_a_entry, 0, 1)
        vector_form_layout.addWidget(self.vec_b_label, 1, 0)
        vector_form_layout.addWidget(self.vec_b_entry, 1, 1)
        vector_form_layout.addWidget(self.scalar_v_label, 2, 0)
        vector_form_layout.addWidget(self.scalar_v_entry, 2, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        vector_groupbox.setLayout(vector_form_layout)
        left_column_layout.addWidget(vector_groupbox)

        # --- Sección Parámetros Métodos Iterativos ---
        iter_params_groupbox = QGroupBox("Parámetros Métodos Iterativos")
        iter_params_groupbox.setFont(self.fonts["group_title"])
        iter_params_groupbox.setStyleSheet(self._get_groupbox_style())
        iter_params_layout = QGridLayout()
        iter_params_layout.setSpacing(10)

        self.tol_label = QLabel("Tolerancia (ej: 1e-6):")
        self.tol_label.setFont(self.fonts["label"])
        self.tol_label.setStyleSheet(f"color: {self.colors['text_label']};")
        self.tol_entry = QLineEdit("1e-6") 
        self.tol_entry.setFont(self.fonts["entry"])
        self.tol_entry.setStyleSheet(self._get_lineedit_style())
        self.tol_entry.setFixedWidth(180)

        self.max_iter_label = QLabel("Max. Iteraciones (ej: 1000):")
        self.max_iter_label.setFont(self.fonts["label"])
        self.max_iter_label.setStyleSheet(f"color: {self.colors['text_label']};")
        self.max_iter_entry = QLineEdit("1000") 
        self.max_iter_entry.setFont(self.fonts["entry"])
        self.max_iter_entry.setStyleSheet(self._get_lineedit_style())
        self.max_iter_entry.setFixedWidth(180)

        self.x0_label = QLabel("Aprox. Inicial x0 (ej: 0,0,0):")
        self.x0_label.setFont(self.fonts["label"])
        self.x0_label.setStyleSheet(f"color: {self.colors['text_label']};")
        self.x0_entry = QLineEdit() 
        self.x0_entry.setPlaceholderText("Vacío = vector de ceros")
        self.x0_entry.setFont(self.fonts["entry"])
        self.x0_entry.setStyleSheet(self._get_lineedit_style())

        self.omega_label = QLabel("Omega (SOR, 0<ω<2):")
        self.omega_label.setFont(self.fonts["label"])
        self.omega_label.setStyleSheet(f"color: {self.colors['text_label']};")
        self.omega_entry = QLineEdit("1.1") 
        self.omega_entry.setFont(self.fonts["entry"])
        self.omega_entry.setStyleSheet(self._get_lineedit_style())
        self.omega_entry.setFixedWidth(180)

        iter_params_layout.addWidget(self.tol_label, 0, 0)
        iter_params_layout.addWidget(self.tol_entry, 0, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        iter_params_layout.addWidget(self.max_iter_label, 1, 0)
        iter_params_layout.addWidget(self.max_iter_entry, 1, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        iter_params_layout.addWidget(self.x0_label, 2, 0)
        iter_params_layout.addWidget(self.x0_entry, 2, 1)
        iter_params_layout.addWidget(self.omega_label, 3, 0)
        iter_params_layout.addWidget(self.omega_entry, 3, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        iter_params_groupbox.setLayout(iter_params_layout)
        left_column_layout.addWidget(iter_params_groupbox)
        left_column_layout.addStretch(1) 

        # --- Columna Derecha (Matrices) ---
        right_column_layout = QVBoxLayout()
        right_column_layout.setSpacing(15)

        # --- Sección de Matrices ---
        matrix_groupbox = QGroupBox("Matrices")
        matrix_groupbox.setFont(self.fonts["group_title"])
        matrix_groupbox.setStyleSheet(self._get_groupbox_style()) 
        matrix_form_layout = QGridLayout()
        matrix_form_layout.setSpacing(10)

        self.mat_a_label = QLabel("Matriz A (filas con ';', elems con ','):")
        self.mat_a_label.setFont(self.fonts["label"])
        self.mat_a_label.setStyleSheet(f"color: {self.colors['text_label']};") # Color explícito
        self.mat_a_text = QTextEdit()
        self.mat_a_text.setFont(self.fonts["entry"])
        self.mat_a_text.setMinimumHeight(100) 
        self.mat_a_text.setStyleSheet(self._get_textedit_style())

        self.mat_b_label = QLabel("Matriz B (filas con ';', elems con ','):")
        self.mat_b_label.setFont(self.fonts["label"])
        self.mat_b_label.setStyleSheet(f"color: {self.colors['text_label']};") # Color explícito
        self.mat_b_text = QTextEdit()
        self.mat_b_text.setFont(self.fonts["entry"])
        self.mat_b_text.setMinimumHeight(100)
        self.mat_b_text.setStyleSheet(self._get_textedit_style())

        self.scalar_m_label = QLabel("Escalar (Matrices):")
        self.scalar_m_label.setFont(self.fonts["label"])
        self.scalar_m_label.setStyleSheet(f"color: {self.colors['text_label']};") # Color explícito
        self.scalar_m_entry = QLineEdit()
        self.scalar_m_entry.setFont(self.fonts["entry"])
        self.scalar_m_entry.setStyleSheet(self._get_lineedit_style())
        self.scalar_m_entry.setFixedWidth(180)

        matrix_form_layout.addWidget(self.mat_a_label, 0, 0, alignment=Qt.AlignmentFlag.AlignTop)
        matrix_form_layout.addWidget(self.mat_a_text, 0, 1)
        matrix_form_layout.addWidget(self.mat_b_label, 1, 0, alignment=Qt.AlignmentFlag.AlignTop)
        matrix_form_layout.addWidget(self.mat_b_text, 1, 1)
        matrix_form_layout.addWidget(self.scalar_m_label, 2, 0)
        matrix_form_layout.addWidget(self.scalar_m_entry, 2, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        matrix_form_layout.setColumnStretch(1, 1)
        matrix_groupbox.setLayout(matrix_form_layout)
        right_column_layout.addWidget(matrix_groupbox)
        right_column_layout.addStretch(1)

        # Añadir columnas al layout de entrada
        input_layout.addLayout(left_column_layout, stretch=1)
        input_layout.addLayout(right_column_layout, stretch=2) 
        main_layout.addWidget(input_frame)


        # --- Sección de Operaciones (Botones) ---
        ops_groupbox = QGroupBox("Operaciones a Realizar")
        ops_groupbox.setFont(self.fonts["group_title"])
        ops_groupbox.setStyleSheet(self._get_groupbox_style())
        ops_layout = QGridLayout()
        ops_layout.setSpacing(10)
        ops_layout.setContentsMargins(15, 20, 15, 15) 

        # Botones de Vectores
        vec_buttons_layout = QGridLayout()
        vec_buttons_layout.setSpacing(8)
        vec_buttons_data = [
            ("A + B", lambda: self.perform_vector_op('add')), ("A - B", lambda: self.perform_vector_op('subtract')),
            ("Producto Punto", lambda: self.perform_vector_op('dot')), ("Producto Cruz", lambda: self.perform_vector_op('cross')), 
            ("Escalar * A", lambda: self.perform_vector_op('scalar_mult_a')), ("Escalar * B", lambda: self.perform_vector_op('scalar_mult_b')),
            ("Magnitud A", lambda: self.perform_vector_op('magnitude_a')), ("Magnitud B", lambda: self.perform_vector_op('magnitude_b')), 
        ]
        buttons_per_row_vec = 4 
        for i, (text, func) in enumerate(vec_buttons_data):
            btn = QPushButton(text)
            btn.setFont(self.fonts["button"])
            btn.setStyleSheet(self._get_button_style())
            btn.clicked.connect(func)
            vec_buttons_layout.addWidget(btn, i // buttons_per_row_vec, i % buttons_per_row_vec)
        
        # Botones de Matrices (Básicos y Análisis)
        mat_buttons_layout_1 = QGridLayout()
        mat_buttons_layout_1.setSpacing(8)
        mat_buttons_data_1 = [
            ("A + B", lambda: self.perform_matrix_op('add')),("A - B", lambda: self.perform_matrix_op('subtract')),
            ("A * B", lambda: self.perform_matrix_op('multiply')),("Escalar * A", lambda: self.perform_matrix_op('scalar_mult_a')),
            ("Escalar * B", lambda: self.perform_matrix_op('scalar_mult_b')),("Transpuesta A", lambda: self.perform_matrix_op('transpose_a')), 
            ("Transpuesta B", lambda: self.perform_matrix_op('transpose_b')),("det(A)", lambda: self.perform_matrix_op('determinant_a')),
            ("det(B)", lambda: self.perform_matrix_op('determinant_b')),("Inversa A", lambda: self.perform_matrix_op('inverse_a')), 
            ("Inversa B", lambda: self.perform_matrix_op('inverse_b')),("Condición A", lambda: self.perform_matrix_op('condition_a')), 
            ("Condición B", lambda: self.perform_matrix_op('condition_b')),("Val/Vec Propios A", lambda: self.perform_matrix_op('eigen_a')), 
            ("Val/Vec Propios B", lambda: self.perform_matrix_op('eigen_b')),("Dom. Diagonal A", lambda: self.perform_matrix_op('diag_dominance_a')), 
            ("Dom. Diagonal B", lambda: self.perform_matrix_op('diag_dominance_b')),
        ]
        buttons_per_row_mat1 = 4 
        for i, (text, func) in enumerate(mat_buttons_data_1):
            btn = QPushButton(text)
            btn.setFont(self.fonts["button"])
            btn.setStyleSheet(self._get_button_style())
            btn.clicked.connect(func)
            mat_buttons_layout_1.addWidget(btn, i // buttons_per_row_mat1, i % buttons_per_row_mat1)

        # Botones de Solución de Sistemas
        solve_buttons_layout = QGridLayout()
        solve_buttons_layout.setSpacing(8)
        solve_buttons_data = [
            ("Gauss", lambda: self.perform_matrix_op('gauss')),
            ("Gauss-Jordan", lambda: self.perform_matrix_op('gauss_jordan')),
            ("LU Solve", lambda: self.perform_matrix_op('lu_doolittle_solve')), 
            ("Cholesky Solve", lambda: self.perform_matrix_op('cholesky_solve')), 
            ("Jacobi", lambda: self.perform_matrix_op('jacobi')),
            ("Gauss-Seidel", lambda: self.perform_matrix_op('gauss_seidel')),
            ("SOR", lambda: self.perform_matrix_op('sor')),
            ("Grad. Conjugado", lambda: self.perform_matrix_op('conjugate_gradient')),
            ("Desc. Pronunciado", lambda: self.perform_matrix_op('steepest_descent')), 
        ]
        buttons_per_row_solve = 3 
        for i, (text, func) in enumerate(solve_buttons_data):
            btn = QPushButton(text)
            btn.setFont(self.fonts["button"])
            btn.setStyleSheet(self._get_button_style())
            btn.clicked.connect(func)
            solve_buttons_layout.addWidget(btn, i // buttons_per_row_solve, i % buttons_per_row_solve)
            
        # Botón Problema Específico
        specific_problem_btn = QPushButton("Ejecutar Problema Específico (n=14..17)")
        specific_problem_btn.setFont(self.fonts["button"])
        specific_problem_btn.setStyleSheet(self._get_button_style(clear=False)) 
        specific_problem_btn.clicked.connect(self.run_specific_problem_gui)

        # Botones Limpiar
        clear_buttons_layout = QHBoxLayout()
        clear_buttons_layout.setSpacing(15)
        self.clear_vec_btn = QPushButton("Limpiar Vectores")
        self.clear_vec_btn.setFont(self.fonts["button"])
        self.clear_vec_btn.setStyleSheet(self._get_button_style(clear=True))
        self.clear_vec_btn.clicked.connect(self.clear_vector_fields)
        
        self.clear_mat_btn = QPushButton("Limpiar Matrices")
        self.clear_mat_btn.setFont(self.fonts["button"])
        self.clear_mat_btn.setStyleSheet(self._get_button_style(clear=True))
        self.clear_mat_btn.clicked.connect(self.clear_matrix_fields)

        self.clear_all_btn = QPushButton("Limpiar Todo")
        self.clear_all_btn.setFont(self.fonts["button"])
        self.clear_all_btn.setStyleSheet(self._get_button_style(clear=True))
        self.clear_all_btn.clicked.connect(self.clear_all_fields)
        
        clear_buttons_layout.addWidget(self.clear_vec_btn)
        clear_buttons_layout.addWidget(self.clear_mat_btn)
        clear_buttons_layout.addWidget(self.clear_all_btn)
        clear_buttons_layout.addStretch(1)


        # Añadir layouts de botones al layout principal de operaciones
        ops_layout.addWidget(QLabel("<b>Vectores:</b>"), 0, 0, 1, buttons_per_row_vec) 
        ops_layout.addLayout(vec_buttons_layout, 1, 0, 1, buttons_per_row_vec)
        ops_layout.addWidget(QLabel("<b>Matrices (Análisis y Ops Básicas):</b>"), 2, 0, 1, buttons_per_row_mat1)
        ops_layout.addLayout(mat_buttons_layout_1, 3, 0, 1, buttons_per_row_mat1)
        ops_layout.addWidget(QLabel("<b>Solución Sistemas Ax=b:</b>"), 4, 0, 1, buttons_per_row_solve)
        ops_layout.addLayout(solve_buttons_layout, 5, 0, 1, buttons_per_row_solve)
        ops_layout.addWidget(specific_problem_btn, 6, 0, 1, 4) 
        ops_layout.addLayout(clear_buttons_layout, 7, 0, 1, 4) 

        max_cols = max(buttons_per_row_vec, buttons_per_row_mat1, buttons_per_row_solve)
        for i in range(max_cols):
             ops_layout.setColumnStretch(i, 1)

        ops_groupbox.setLayout(ops_layout)
        main_layout.addWidget(ops_groupbox)


        # --- Área de Resultados ---
        result_groupbox = QGroupBox("Resultado")
        result_groupbox.setFont(self.fonts["group_title"])
        result_groupbox.setStyleSheet(self._get_groupbox_style()) 
        result_layout = QVBoxLayout()
        
        self.result_text = QTextEdit()
        self.result_text.setFont(self.fonts["result"])
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {self.colors['result_bg']};
                color: {self.colors['result_fg']};
                border: 1px solid {self.colors['border']};
                border-radius: 5px;
                padding: 10px;
            }}
        """)
        result_layout.addWidget(self.result_text)
        result_groupbox.setLayout(result_layout)
        main_layout.addWidget(result_groupbox, stretch=1) 

        # --- Establecer el layout principal en el widget contenedor del scroll ---
        scroll_content_widget.setLayout(main_layout)
        
        # --- Configurar el ScrollArea ---
        scroll_area.setWidget(scroll_content_widget)
        
        # --- Layout final de la ventana principal ---
        window_layout = QVBoxLayout(self) 
        window_layout.setContentsMargins(0,0,0,0) 
        window_layout.addWidget(scroll_area)
        self.setLayout(window_layout) 

    # --- Funciones de Estilo ---
    def _get_groupbox_style(self):
        # Ajustar padding-top y margin-top
        return f"""
            QGroupBox {{ 
                background-color: {self.colors['bg_groupbox']}; 
                border: 1px solid {self.colors['border']}; 
                border-radius: 8px; 
                margin-top: 15px; /* Aumentar margen superior */
                padding-top: 30px; /* Aumentar padding para que el título no solape */
                padding-bottom: 15px;
                padding-left: 15px;
                padding-right: 15px;
            }}
            QGroupBox::title {{ 
                subcontrol-origin: margin; 
                subcontrol-position: top left; 
                padding: 5px 15px; 
                margin-left: 10px; 
                color: {self.colors['groupbox_title_fg']};
                background-color: {self.colors['bg_groupbox']}; 
                border-radius: 4px; 
            }}
        """

    def _get_lineedit_style(self):
        # ... (sin cambios) ...
        return f"""
            QLineEdit {{
                background-color: {self.colors['bg_entry']};
                border: 1px solid {self.colors['border']};
                padding: 7px; 
                border-radius: 5px; 
                color: {self.colors['text_main']};
            }}
            QLineEdit:focus {{
                border: 2px solid {self.colors['button_bg']}; 
            }}
        """
    
    def _get_textedit_style(self):
        # ... (sin cambios) ...
        return f"""
            QTextEdit {{
                background-color: {self.colors['bg_entry']};
                border: 1px solid {self.colors['border']};
                padding: 7px;
                border-radius: 5px;
                color: {self.colors['text_main']};
            }}
            QTextEdit:focus {{
                border: 2px solid {self.colors['button_bg']};
            }}
        """

    def _get_button_style(self, clear=False):
        # ... (sin cambios) ...
        bg = self.colors['clear_button_bg'] if clear else self.colors['button_bg']
        hover_bg = self.colors['clear_button_hover_bg'] if clear else self.colors['button_hover_bg']
        return f"""
            QPushButton {{
                background-color: {bg};
                color: {self.colors['button_fg']};
                border: none; 
                padding: 9px 10px; 
                border-radius: 5px; 
                min-height: 22px; 
            }}
            QPushButton:hover {{
                background-color: {hover_bg};
            }}
            QPushButton:pressed {{
                background-color: {hover_bg}; 
                padding-top: 10px; 
                padding-bottom: 8px;
            }}
        """

    # --- Funciones de Parseo (sin cambios) ---
    def _parse_vector(self, vec_str):
        # ... (código idéntico) ...
        if not vec_str.strip():
            return None
        try:
            vec_str = vec_str.replace(' ', ',') 
            parts = [p for p in vec_str.split(',') if p.strip()] 
            return np.array([float(x.strip()) for x in parts])
        except ValueError:
            self.show_error("Error de formato en vector. Use números separados por comas (o espacios).")
            return None

    def _parse_matrix(self, mat_str):
        # ... (código idéntico) ...
        if not mat_str.strip():
            return None
        try:
            rows = mat_str.split(';')
            matrix = []
            first_row_len = -1
            processed_rows = 0
            for i, row_str in enumerate(rows):
                clean_row_str = row_str.strip()
                if not clean_row_str: continue 
                
                clean_row_str = clean_row_str.replace(' ', ',')
                row_parts = [p for p in clean_row_str.split(',') if p.strip()]
                if not row_parts: continue 

                row = [float(x.strip()) for x in row_parts]
                
                if processed_rows == 0: 
                    first_row_len = len(row)
                
                if first_row_len != -1 and len(row) != first_row_len: 
                    self.show_error("Error de formato en matriz: Todas las filas deben tener la misma cantidad de elementos.")
                    return None
                
                matrix.append(row)
                processed_rows += 1
            
            if not matrix: 
                return None
            return np.array(matrix)
        except (ValueError, IndexError): 
            self.show_error("Error de formato en matriz. Use números sep. por comas/espacios y filas por punto y coma (ej: 1,2;3,4).")
            return None

    def _get_scalar(self, entry_widget):
        # ... (código idéntico) ...
        scalar_str = entry_widget.text()
        if not scalar_str.strip():
            return None 
        try:
            return float(scalar_str)
        except ValueError:
            self.show_error("Escalar inválido. Debe ser un número.")
            return None 

    def _get_iter_params(self):
        # ... (código idéntico) ...
        try:
            tol = float(self.tol_entry.text())
            if tol <= 0: raise ValueError("La tolerancia debe ser positiva.")
        except ValueError:
            self.show_error("Tolerancia inválida. Usando 1e-6.")
            tol = 1e-6
            self.tol_entry.setText("1e-6")

        try:
            max_iter = int(self.max_iter_entry.text())
            if max_iter <= 0: raise ValueError("Máx. iteraciones debe ser positivo.")
        except ValueError:
            self.show_error("Máx. iteraciones inválido. Usando 1000.")
            max_iter = 1000
            self.max_iter_entry.setText("1000")
        
        x0 = self._parse_vector(self.x0_entry.text()) 

        try:
            omega = float(self.omega_entry.text())
            if not (0 < omega < 2):
                 QMessageBox.warning(self, "Advertencia Omega", 
                                     f"Omega = {omega} está fuera del rango (0, 2) recomendado para convergencia SOR.")
        except ValueError:
            self.show_error("Omega inválido. Usando 1.1.")
            omega = 1.1
            self.omega_entry.setText("1.1")

        return tol, max_iter, x0, omega

    # --- Funciones de Visualización y Limpieza (sin cambios lógicos) ---
    def show_result(self, result_data): 
        # ... (código idéntico) ...
        self.result_text.clear()
        
        output_str = ""
        # Caso especial para salida de métodos iterativos
        if isinstance(result_data, dict) and 'solution' in result_data:
            sol = result_data['solution']
            iters = result_data['iterations']
            converged = result_data['converged']
            ops = result_data.get('ops', {}) 
            method = result_data.get('method', 'Método Iterativo')
            time_taken = result_data.get('time', None) 
            
            output_str += f"<b>Resultado ({method}):</b>\n"
            if converged:
                output_str += f"Convergió en {iters} iteraciones.\n"
            else:
                output_str += f"<font color='{self.colors['clear_button_bg']}'><b>NO convergió</b></font> después de {iters} iteraciones.\n"
            
            if time_taken is not None:
                 output_str += f"Tiempo de ejecución: {time_taken:.6f} segundos.\n"

            output_str += "Solución x (o última aproximación):\n"
            if isinstance(sol, np.ndarray):
                 res_str = np.array2string(sol, precision=6, suppress_small=True, separator=', ', 
                                          threshold=np.inf, edgeitems=10, 
                                          formatter={'float_kind':lambda x: "%.6f" % x}) 
                 if sol.ndim == 1: res_str = res_str.replace('[[', '[').replace(']]', ']')
                 output_str += res_str
                 if np.isnan(sol).any():
                    output_str += "\n<font color='orange'><b>Advertencia:</b> La solución contiene valores NaN.</font>"
            else:
                 output_str += str(sol) 

            if ops:
                 output_str += "\n\n<b>Operaciones (Estimación):</b>\n"
                 keys_ordered = sorted([k for k, v in ops.items() if v > 0])
                 for key in keys_ordered:
                     if key in ['matmul', 'dot', 'norm']:
                         output_str += f"- {key.capitalize()} (Llamadas NumPy): {ops[key]}\n"
                     else:
                         output_str += f"- {key.replace('_',' ').capitalize()}: {ops[key]}\n"
            
        # Caso para tuplas (descripción, datos)
        elif isinstance(result_data, tuple) and len(result_data) == 2 and isinstance(result_data[0], str): 
            header, data = result_data
            output_str += f"<b>{header}</b>\n" 
            if isinstance(data, np.ndarray):
                res_str = np.array2string(data, precision=4, suppress_small=True, separator=', ', 
                                          threshold=np.inf, edgeitems=10, 
                                          formatter={'float_kind':lambda x: "%.4f" % x}) 
                if data.ndim == 1: res_str = res_str.replace('[[', '[').replace(']]', ']')
                output_str += res_str
            elif isinstance(data, str) and data != "": 
                 output_str += data 
            elif isinstance(data, (int, float)):
                 output_str += f"{data:.4e}" 
            elif data is not None and data != "": 
                output_str += str(data)
        # Caso para arrays NumPy directos
        elif isinstance(result_data, np.ndarray):
            res_str = np.array2string(result_data, precision=4, suppress_small=True, separator=', ', 
                                      threshold=np.inf, edgeitems=10,
                                      formatter={'float_kind':lambda x: "%.4f" % x})
            if result_data.ndim == 1: res_str = res_str.replace('[[', '[').replace(']]', ']')
            output_str += res_str
        # Otros casos (escalares, strings simples)
        elif result_data is not None:
             if isinstance(result_data, (int, float)):
                 if abs(result_data) > 1e5 or (abs(result_data) < 1e-4 and result_data != 0):
                     output_str += f"{result_data:.4e}" 
                 else:
                     output_str += f"{result_data:.6f}" 
             else:
                 output_str += str(result_data)
        
        self.result_text.setHtml(output_str.replace("\n", "<br>")) 

    def show_error(self, message):
        # ... (código idéntico) ...
        QMessageBox.critical(self, "Error", message) 
        self.result_text.setHtml(f"<font color='{self.colors['clear_button_bg']}'><b>Error:</b> {message}</font>") 

    def clear_vector_fields(self):
        # ... (código idéntico) ...
        self.vec_a_entry.clear()
        self.vec_b_entry.clear()
        self.scalar_v_entry.clear()
        self.x0_entry.clear() 
        self.show_result("<i>Campos de vectores y x0 limpiados.</i>") 

    def clear_matrix_fields(self):
        # ... (código idéntico) ...
        self.mat_a_text.clear()
        self.mat_b_text.clear()
        self.scalar_m_entry.clear()
        self.L = None
        self.U = None
        self.L_chol = None
        self.show_result("<i>Campos de matrices y factorizaciones limpiados.</i>")
        
    def clear_all_fields(self):
        # ... (código idéntico) ...
        self.clear_vector_fields()
        self.clear_matrix_fields()
        self.tol_entry.setText("1e-6")
        self.max_iter_entry.setText("1000")
        self.omega_entry.setText("1.1")
        self.result_text.clear() 
        self.show_result("<i>Todos los campos limpiados y parámetros reseteados.</i>")

    # --- Lógica de Operaciones ---
    def perform_vector_op(self, operation):
        # ... (código idéntico) ...
        vec_a = self._parse_vector(self.vec_a_entry.text())
        vec_b = self._parse_vector(self.vec_b_entry.text())
        scalar = self._get_scalar(self.scalar_v_entry)
        result = None

        try:
            if operation == 'add':
                if vec_a is None or vec_b is None: raise ValueError("Ambos vectores A y B son requeridos.")
                if vec_a.shape != vec_b.shape: raise ValueError("Vectores deben tener la misma dimensión para sumar.")
                result = vec_a + vec_b
            elif operation == 'subtract':
                if vec_a is None or vec_b is None: raise ValueError("Ambos vectores A y B son requeridos.")
                if vec_a.shape != vec_b.shape: raise ValueError("Vectores deben tener la misma dimensión para restar.")
                result = vec_a - vec_b
            elif operation == 'dot':
                if vec_a is None or vec_b is None: raise ValueError("Ambos vectores A y B son requeridos.")
                if vec_a.shape != vec_b.shape: raise ValueError("Vectores deben tener la misma dimensión para producto punto.")
                result = np.dot(vec_a, vec_b)
            elif operation == 'cross':
                if vec_a is None or vec_b is None: raise ValueError("Ambos vectores A y B son requeridos.")
                if vec_a.size != 3 or vec_b.size != 3: raise ValueError("Producto cruz definido solo para vectores 3D.")
                result = np.cross(vec_a, vec_b)
            elif operation == 'scalar_mult_a':
                if vec_a is None: raise ValueError("Vector A es requerido.")
                if scalar is None: raise ValueError("Escalar es requerido.")
                result = vec_a * scalar
            elif operation == 'scalar_mult_b':
                if vec_b is None: raise ValueError("Vector B es requerido.")
                if scalar is None: raise ValueError("Escalar es requerido.")
                result = vec_b * scalar
            elif operation == 'magnitude_a':
                if vec_a is None: raise ValueError("Vector A es requerido.")
                result = np.linalg.norm(vec_a)
            elif operation == 'magnitude_b':
                if vec_b is None: raise ValueError("Vector B es requerido.")
                result = np.linalg.norm(vec_b)
            
            if result is not None:
                self.show_result(result)

        except ValueError as e:
            self.show_error(str(e))
        except Exception as e:
            self.show_error(f"Error inesperado en operación vectorial: {str(e)}")


    def perform_matrix_op(self, operation):
        # ... (Lógica de parseo y obtención de parámetros idéntica) ...
        current_A_text = self.mat_a_text.toPlainText().strip()
        current_B_text = self.mat_b_text.toPlainText().strip()
        current_b_vec_text = self.vec_b_entry.text().strip()
        
        parsed_A = self._parse_matrix(current_A_text)
        if parsed_A is not None: self.A = parsed_A
        
        parsed_b = self._parse_vector(current_b_vec_text)
        if parsed_b is not None: self.b = parsed_b
            
        mat_b_gui = self._parse_matrix(current_B_text) 
        scalar = self._get_scalar(self.scalar_m_entry)
        result_data = None 
        start_time = time.time() 

        try:
            # --- Operaciones Básicas y Análisis (sin cambios) ---
            if operation == 'add':
                if self.A is None or mat_b_gui is None: raise ValueError("Ambas matrices A y B son requeridas.")
                if self.A.shape != mat_b_gui.shape: raise ValueError("Matrices deben tener las mismas dimensiones para sumar.")
                result_data = self.A + mat_b_gui
            elif operation == 'subtract':
                if self.A is None or mat_b_gui is None: raise ValueError("Ambas matrices A y B son requeridas.")
                if self.A.shape != mat_b_gui.shape: raise ValueError("Matrices deben tener las mismas dimensiones para restar.")
                result_data = self.A - mat_b_gui
            # ... (resto de operaciones básicas y análisis idénticas) ...
            elif operation == 'multiply':
                if self.A is None or mat_b_gui is None: raise ValueError("Ambas matrices A y B son requeridas.")
                if self.A.shape[1] != mat_b_gui.shape[0]: raise ValueError("Columnas de A deben ser igual a filas de B para multiplicar.")
                result_data = np.matmul(self.A, mat_b_gui)
            elif operation == 'scalar_mult_a':
                if self.A is None: raise ValueError("Matriz A es requerida.")
                if scalar is None: raise ValueError("Escalar es requerido.")
                result_data = self.A * scalar
            elif operation == 'scalar_mult_b':
                if mat_b_gui is None: raise ValueError("Matriz B es requerida.")
                if scalar is None: raise ValueError("Escalar es requerido.")
                result_data = mat_b_gui * scalar
            elif operation == 'transpose_a':
                if self.A is None: raise ValueError("Matriz A es requerida.")
                result_data = self.A.T
            elif operation == 'transpose_b':
                if mat_b_gui is None: raise ValueError("Matriz B es requerida.")
                result_data = mat_b_gui.T
            elif operation == 'determinant_a':
                if self.A is None: raise ValueError("Matriz A es requerida.")
                if self.A.shape[0] != self.A.shape[1]: raise ValueError("Matriz A debe ser cuadrada para determinante.")
                result_data = np.linalg.det(self.A)
            elif operation == 'determinant_b':
                if mat_b_gui is None: raise ValueError("Matriz B es requerida.")
                if mat_b_gui.shape[0] != mat_b_gui.shape[1]: raise ValueError("Matriz B debe ser cuadrada para determinante.")
                result_data = np.linalg.det(mat_b_gui)
            elif operation == 'inverse_a':
                if self.A is None: raise ValueError("Matriz A es requerida.")
                if self.A.shape[0] != self.A.shape[1]: raise ValueError("Matriz A debe ser cuadrada para inversa.")
                try: result_data = np.linalg.inv(self.A)
                except np.linalg.LinAlgError: raise ValueError("Matriz A es singular, no se puede invertir.")
            elif operation == 'inverse_b':
                if mat_b_gui is None: raise ValueError("Matriz B es requerida.")
                if mat_b_gui.shape[0] != mat_b_gui.shape[1]: raise ValueError("Matriz B debe ser cuadrada para inversa.")
                try: result_data = np.linalg.inv(mat_b_gui)
                except np.linalg.LinAlgError: raise ValueError("Matriz B es singular, no se puede invertir.")
            elif operation == 'eigen_a':
                if self.A is None: raise ValueError("Matriz A es requerida.")
                if self.A.shape[0] != self.A.shape[1]: raise ValueError("Matriz A debe ser cuadrada.")
                eigenvalues, eigenvectors = np.linalg.eig(self.A)
                result_data = (f"Valores Propios A:\n{np.array2string(eigenvalues, precision=4, suppress_small=True, formatter={'float_kind':lambda x: '%.4f' % x})}\n\nVectores Propios A (columnas):\n", eigenvectors)
            elif operation == 'eigen_b':
                if mat_b_gui is None: raise ValueError("Matriz B es requerida.")
                if mat_b_gui.shape[0] != mat_b_gui.shape[1]: raise ValueError("Matriz B debe ser cuadrada.")
                eigenvalues, eigenvectors = np.linalg.eig(mat_b_gui)
                result_data = (f"Valores Propios B:\n{np.array2string(eigenvalues, precision=4, suppress_small=True, formatter={'float_kind':lambda x: '%.4f' % x})}\n\nVectores Propios B (columnas):\n", eigenvectors)
            elif operation == 'condition_a':
                if self.A is None: raise ValueError("Matriz A es requerida.")
                cond_num = np.linalg.cond(self.A)
                result_data = (f"Número Condición A (norma 2): {cond_num:.4e}", "") 
            elif operation == 'condition_b':
                if mat_b_gui is None: raise ValueError("Matriz B es requerida.")
                cond_num = np.linalg.cond(mat_b_gui)
                result_data = (f"Número Condición B (norma 2): {cond_num:.4e}", "")
            elif operation == 'diag_dominance_a':
                if self.A is None: raise ValueError("Matriz A es requerida.")
                if self.A.shape[0] != self.A.shape[1]: raise ValueError("Matriz A debe ser cuadrada.")
                summary, details = self.check_diagonal_dominance(self.A)
                result_data = (summary, details) 
            elif operation == 'diag_dominance_b':
                if mat_b_gui is None: raise ValueError("Matriz B es requerida.")
                if mat_b_gui.shape[0] != mat_b_gui.shape[1]: raise ValueError("Matriz B debe ser cuadrada.")
                summary, details = self.check_diagonal_dominance(mat_b_gui)
                result_data = (summary, details)

            # --- Métodos de Solución Ax=b ---
            elif operation in ['gauss', 'gauss_jordan', 'lu_doolittle_solve', 'cholesky_solve', 
                               'jacobi', 'gauss_seidel', 'sor', 'conjugate_gradient', 
                               'steepest_descent']: 
                
                # ... (Validaciones comunes idénticas) ...
                if self.A is None: raise ValueError("Matriz A es requerida para Ax=b.")
                if self.b is None: raise ValueError("Vector b es requerido para Ax=b.")
                if self.A.shape[0] != self.A.shape[1]: raise ValueError("Matriz A debe ser cuadrada para Ax=b.")
                if self.A.shape[0] != self.b.shape[0]: raise ValueError("Dimensiones de A y b no compatibles.")
                if np.isnan(self.A).any() or np.isnan(self.b).any(): raise ValueError("Matriz A o vector b contienen NaN.")

                tol, max_iter, x0, omega = None, None, None, None
                is_iterative = operation in ['jacobi', 'gauss_seidel', 'sor', 'conjugate_gradient', 'steepest_descent'] 
                if is_iterative:
                    tol, max_iter, x0, omega = self._get_iter_params()
                    if x0 is not None and x0.shape[0] != self.A.shape[0]:
                        self.show_error(f"Tamaño de x0 ({x0.shape[0]}) no coincide con n ({self.A.shape[0]}). Usando ceros.")
                        x0 = None 
                
                solution, ops, iters, converged = None, {}, 0, False 
                method_name_full = operation.replace("_"," ").title()
                # ... (Ajuste de nombres idéntico) ...
                if operation == 'lu_doolittle_solve': method_name_full = "LU Doolittle + Solve"
                if operation == 'cholesky_solve': method_name_full = "Cholesky + Solve"
                if operation == 'sor': method_name_full = f"SOR (ω={omega})"
                if operation == 'conjugate_gradient': method_name_full = "Gradiente Conjugado"
                if operation == 'steepest_descent': method_name_full = "Descenso Pronunciado" 

                try:
                    A_copy, b_copy = self.A.copy(), self.b.copy()
                    
                    if operation == 'gauss':
                        solution, ops = methods.gaussian_elimination(A_copy, b_copy)
                        converged, iters = True, 1
                    elif operation == 'gauss_jordan':
                        solution, ops = methods.gauss_jordan_elimination(A_copy, b_copy)
                        converged, iters = True, 1
                    elif operation == 'lu_doolittle_solve':
                        # ... (lógica LU idéntica) ...
                        if self.L is None or self.U is None:
                            print("Calculando factorización Doolittle...") 
                            self.L, self.U, ops_fact = methods.doolittle_factorization(A_copy)
                            print("Factorización completada.")
                        else:
                            print("Usando factorización LU Doolittle guardada.")
                            ops_fact = {} 
                        solution, ops_solve = methods.lu_solve(self.L, self.U, b_copy)
                        ops = {k: ops_fact.get(k, 0) + ops_solve.get(k, 0) for k in set(ops_fact) | set(ops_solve)}
                        converged, iters = True, 1
                    elif operation == 'cholesky_solve':
                         # ... (lógica Cholesky idéntica) ...
                         if self.L_chol is None:
                            print("Calculando factorización Cholesky...")
                            if not methods.es_simetrica(A_copy): raise ValueError("Cholesky requiere matriz simétrica.")
                            self.L_chol, ops_fact = methods.cholesky_factorization(A_copy)
                            print("Factorización completada.")
                         else:
                            print("Usando factorización Cholesky guardada.")
                            ops_fact = {}
                         solution, ops_solve = methods.cholesky_solve(self.L_chol, b_copy)
                         ops = {k: ops_fact.get(k, 0) + ops_solve.get(k, 0) for k in set(ops_fact) | set(ops_solve)}
                         converged, iters = True, 1
                    elif operation == 'jacobi':
                        solution, iters, converged, ops = methods.jacobi_method(A_copy, b_copy, x0, tol, max_iter)
                    elif operation == 'gauss_seidel':
                        solution, iters, converged, ops = methods.gauss_seidel_method(A_copy, b_copy, x0, tol, max_iter)
                    elif operation == 'sor':
                        solution, iters, converged, ops = methods.sor_method(A_copy, b_copy, omega, x0, tol, max_iter)
                    elif operation == 'conjugate_gradient':
                         solution, iters, converged, ops = methods.conjugate_gradient(A_copy, b_copy, x0, tol, max_iter)
                    elif operation == 'steepest_descent': # <-- LLAMADA A STEEPEST DESCENT
                         solution, iters, converged, ops = steepest_descent(A_copy, b_copy, x0, tol, max_iter)


                    end_time = time.time() 
                    result_data = {
                        'method': method_name_full, 'solution': solution, 
                        'iterations': iters, 'converged': converged, 
                        'ops': ops, 'time': end_time - start_time 
                    }

                except (ValueError, np.linalg.LinAlgError) as e_solve:
                    self.show_error(f"Error en {method_name_full}: {e_solve}")
                    return 

            else: 
                raise ValueError(f"Operación desconocida: {operation}")

            # --- Mostrar resultado ---
            # ... (lógica de show_result idéntica) ...
            if result_data is not None and operation not in ['gauss', 'gauss_jordan', 'lu_doolittle_solve', 'cholesky_solve', 
                                                             'jacobi', 'gauss_seidel', 'sor', 'conjugate_gradient', 'steepest_descent']:
                 end_time = time.time()
                 print(f"Tiempo de ejecución para '{operation}': {end_time - start_time:.6f} segundos")
                 self.show_result(result_data)
            elif result_data is not None: 
                 self.show_result(result_data)


        except ValueError as e:
            self.show_error(str(e))
        except np.linalg.LinAlgError as e: 
            self.show_error(f"Error de Álgebra Lineal (NumPy): {str(e)}")
        except Exception as e:
            import traceback
            print("--- ERROR INESPERADO ---")
            traceback.print_exc() 
            print("-----------------------")
            self.show_error(f"Error inesperado: {type(e).__name__} - {str(e)}")

    def run_specific_problem_gui(self):
        # ... (sin cambios) ...
        n_list = [14, 15, 16, 17]
        self.show_result(f"Ejecutando problema específico para n={n_list}...\n"
                         "Los resultados detallados aparecerán en la consola/terminal.")
        QApplication.processEvents() 
        
        reply = QMessageBox.information(self, "Problema Específico", 
                                        f"Se generarán y resolverán sistemas para n={n_list}.\n"
                                        "Esto puede tardar unos segundos o minutos, especialmente para n=17.\n"
                                        "La aplicación podría parecer no responder durante el cálculo.\n\n"
                                        "Los detalles se imprimirán en la consola.\n\n¿Desea continuar?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                                        QMessageBox.StandardButton.Yes)

        if reply == QMessageBox.StandardButton.No:
            self.show_result("Ejecución del problema específico cancelada.")
            return

        try:
            print("\n--- INICIO EJECUCIÓN PROBLEMA ESPECÍFICO (Resultados en Consola) ---")
            start_total_time = time.time()
            # Usar la función del módulo importado
            results = methods.run_specific_problem(n_list) 
            end_total_time = time.time()
            print(f"--- FIN EJECUCIÓN PROBLEMA ESPECÍFICO (Tiempo total: {end_total_time - start_total_time:.2f}s) ---")

            summary_lines = ["<b>Resumen Problema Específico (Consola para detalles):</b>"]
            for n_res, res in results.items():
                 time_str = f"{res['solve_time']:.4f}s" if res['solve_time'] is not None else "N/A"
                 gen_time_str = f"{res['gen_time']:.4f}s" if res['gen_time'] is not None else "N/A"
                 summary_lines.append(f" n={n_res}: Estado='{res['status']}', T.Gen={gen_time_str}, T.Sol={time_str}")
            
            self.result_text.clear()
            self.result_text.setHtml("<br>".join(summary_lines))
            
            self.A, self.b, self.L, self.U, self.L_chol = None, None, None, None, None
            QMessageBox.information(self, "Problema Específico Completado", 
                                    f"Ejecución para n={n_list} finalizada.\nRevise la consola para los detalles y soluciones.\n"
                                    "(Variables A, b y factorizaciones reseteadas en la GUI).")

        except Exception as e:
             self.show_error(f"Error al ejecutar problema específico: {e}")
             import traceback
             print("--- ERROR EN PROBLEMA ESPECÍFICO ---")
             traceback.print_exc()
             print("---------------------------------")

    # --- Función check_diagonal_dominance (sin cambios) ---
    def check_diagonal_dominance(self, matrix):
        # ... (código idéntico) ...
        n = matrix.shape[0]
        strictly_dominant_row_count = 0
        weakly_dominant_row_count = 0 
        non_dominant_row_count = 0
        
        details_list = []

        for i in range(n):
            diag_element_abs = abs(matrix[i, i])
            if n == 1:
                 sum_off_diag_abs = 0.0
            else:
                 sum_off_diag_abs = np.sum(np.abs(matrix[i, :])) - diag_element_abs
            
            row_desc = f"Fila {i+1}: |{matrix[i,i]:.3g}| vs Σ|restantes| = {sum_off_diag_abs:.3g}. "
            if diag_element_abs > sum_off_diag_abs + 1e-12: 
                details_list.append(row_desc + "Estrictamente dominante.")
                strictly_dominant_row_count += 1
            elif np.isclose(diag_element_abs, sum_off_diag_abs): 
                details_list.append(row_desc + "Débilmente dominante (igualdad).")
                weakly_dominant_row_count += 1
            else: 
                details_list.append(row_desc + "NO es dominante.")
                non_dominant_row_count += 1
        
        summary = ""
        if non_dominant_row_count > 0:
            summary = "La matriz NO es diagonalmente dominante (al menos una fila no cumple)."
        elif strictly_dominant_row_count == n:
            summary = "La matriz es ESTRICTAMENTE diagonalmente dominante (todas las filas cumplen estrictamente)."
        elif (strictly_dominant_row_count + weakly_dominant_row_count) == n and strictly_dominant_row_count > 0:
             summary = "La matriz es DÉBILMENTE diagonalmente dominante (cumple en todas las filas, con al menos una fila estricta)."
        elif (strictly_dominant_row_count + weakly_dominant_row_count) == n and strictly_dominant_row_count == 0:
             summary = "La matriz es DÉBILMENTE diagonalmente dominante (cumple en todas las filas, pero ninguna es estricta, solo igualdad)."
        else: 
             summary = "La matriz NO es diagonalmente dominante (combinación no válida encontrada)." 
            
        full_details_str = "\n".join(details_list) 
        return summary, full_details_str 


if __name__ == '__main__':
    # ... (Manejo de error de display idéntico) ...
    try:
        app = QApplication(sys.argv)
        ex = AlgebraLinealPyQtGUI()
        ex.show()
        sys.exit(app.exec())
    except RuntimeError as e: 
        if "display" in str(e).lower() or "QXcbConnection" in str(e) or "xcb" in str(e).lower(): 
            print("Error: No se pudo inicializar la interfaz gráfica.")
            print("Este programa requiere un entorno de escritorio o X11 forwarding si se ejecuta remotamente.")
            print("Asegúrate de que la variable de entorno $DISPLAY esté configurada correctamente.")
            print(f"Detalles del error: {e}")
            sys.exit(1)
        else:
            raise 
    except Exception as e: 
        print(f"Ocurrió un error al iniciar la aplicación: {e}")
        print("Asegúrate de tener PyQt6 instalado correctamente ('pip install PyQt6 numpy') y un entorno gráfico disponible.")
        sys.exit(1)

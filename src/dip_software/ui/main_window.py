import sys
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ..core.state import ImageState
from ..processing import point_ops, spatial_filters, histogram, morphology, segmentation, recognition, compression

class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DIP Studio")
        self.geometry("1150x850")
        
        # Fixed naming collision (changed from self.state)
        self.image_state = ImageState()
        
        # Intercept the window close event to ensure clean exit
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.setup_ui()

    def on_closing(self):
        """Kills background loops and closes the app cleanly."""
        plt.close('all')  # Free up Matplotlib memory
        self.quit()       # Stop the Tkinter mainloop
        self.destroy()    # Destroy the UI widgets
        sys.exit(0)       # Force the Python process to terminate

    def _create_ribbon_group(self, parent, title):
        """Helper method to create an MS Word-style ribbon group."""
        group_frame = ctk.CTkFrame(parent, fg_color="transparent")
        group_frame.pack(side="left", fill="y", padx=5)
        
        content_frame = ctk.CTkFrame(group_frame, fg_color="transparent")
        content_frame.pack(expand=True, fill="both", pady=(5, 2))
        
        title_label = ctk.CTkLabel(group_frame, text=title, text_color="gray", font=("Arial", 11, "bold"))
        title_label.pack(side="bottom")
        
        # Separator line
        separator = ctk.CTkFrame(parent, width=2, fg_color=("gray80", "gray20"))
        separator.pack(side="left", fill="y", pady=10, padx=5)
        
        return content_frame

    def setup_ui(self):
        self.grid_rowconfigure(0, weight=0)  # Ribbon area
        self.grid_rowconfigure(1, weight=1)  # Workspace area
        self.grid_columnconfigure(0, weight=1)
        
        # ==========================================
        # TOP RIBBON (MS Word Style TabView)
        # ==========================================
        self.ribbon = ctk.CTkTabview(self, height=130)
        self.ribbon.grid(row=0, column=0, sticky="new", padx=10, pady=(5, 0))
        
        tab_home = self.ribbon.add("Home & File")
        tab_point = self.ribbon.add("Point Processing")
        tab_filters = self.ribbon.add("Spatial Filters")
        tab_morph = self.ribbon.add("Morphology")
        tab_seg = self.ribbon.add("Segmentation & Features")

        # --- HOME & FILE TAB ---
        grp_actions = self._create_ribbon_group(tab_home, "Actions")
        ctk.CTkButton(grp_actions, text="Load Image", command=self.load_image, height=40, width=110).pack(side="left", padx=5)
        ctk.CTkButton(grp_actions, text="Undo", command=self.undo, fg_color="#a83232", hover_color="#802626", height=40, width=80).pack(side="left", padx=5)
        
        grp_export = self._create_ribbon_group(tab_home, "Export")
        v_export = ctk.CTkFrame(grp_export, fg_color="transparent")
        v_export.pack(side="left", padx=5)
        self.quality_label = ctk.CTkLabel(v_export, text="JPEG Quality: 90%", font=("Arial", 11))
        self.quality_label.pack()
        self.quality_slider = ctk.CTkSlider(v_export, from_=1, to=100, number_of_steps=99, command=self.update_quality_label, width=120)
        self.quality_slider.set(90)
        self.quality_slider.pack()
        ctk.CTkButton(grp_export, text="Save Image", command=self.save_image, fg_color="#2b7a4b", hover_color="#1e5c37", height=40).pack(side="left", padx=10)

        # --- POINT PROCESSING TAB ---
        grp_basic_ops = self._create_ribbon_group(tab_point, "Basic Transforms")
        ctk.CTkButton(grp_basic_ops, text="Grayscale", command=lambda: self.apply_op(point_ops.to_grayscale), height=40).pack(side="left", padx=5)
        ctk.CTkButton(grp_basic_ops, text="Negative", command=lambda: self.apply_op(point_ops.apply_negative), height=40).pack(side="left", padx=5)
        ctk.CTkButton(grp_basic_ops, text="Log Transform", command=lambda: self.apply_op(point_ops.log_transformation), height=40).pack(side="left", padx=5)
        
        grp_gamma = self._create_ribbon_group(tab_point, "Gamma Correction")
        v_gamma = ctk.CTkFrame(grp_gamma, fg_color="transparent")
        v_gamma.pack(side="left", padx=5)
        self.gamma_label = ctk.CTkLabel(v_gamma, text="Gamma: 1.0", font=("Arial", 11))
        self.gamma_label.pack()
        self.gamma_slider = ctk.CTkSlider(v_gamma, from_=0.1, to=3.0, number_of_steps=29, command=self.update_gamma_label, width=120)
        self.gamma_slider.set(1.0)
        self.gamma_slider.pack()
        ctk.CTkButton(grp_gamma, text="Apply", command=self.apply_gamma, height=40, width=70).pack(side="left", padx=10)

        # --- SPATIAL FILTERS TAB ---
        grp_blur = self._create_ribbon_group(tab_filters, "Blur & Smoothing")
        v_blur = ctk.CTkFrame(grp_blur, fg_color="transparent")
        v_blur.pack(side="left", padx=5)
        self.blur_label = ctk.CTkLabel(v_blur, text="Kernel: 5x5", font=("Arial", 11))
        self.blur_label.pack()
        self.blur_slider = ctk.CTkSlider(v_blur, from_=3, to=15, number_of_steps=6, command=self.update_blur_label, width=100)
        self.blur_slider.set(5)
        self.blur_slider.pack()
        ctk.CTkButton(grp_blur, text="Mean", command=self.apply_mean_blur, height=40, width=70).pack(side="left", padx=5)
        ctk.CTkButton(grp_blur, text="Gaussian", command=self.apply_gaussian_blur, height=40, width=70).pack(side="left", padx=5)
        
        grp_edge = self._create_ribbon_group(tab_filters, "Edge Detection")
        ctk.CTkButton(grp_edge, text="Sobel Edge", command=lambda: self.apply_op(spatial_filters.sobel_edge), height=40).pack(side="left", padx=5)
        ctk.CTkButton(grp_edge, text="Laplacian", command=lambda: self.apply_op(spatial_filters.laplacian_edge), height=40).pack(side="left", padx=5)

        # --- MORPHOLOGY TAB ---
        grp_morph = self._create_ribbon_group(tab_morph, "Morphological Operations")
        v_morph = ctk.CTkFrame(grp_morph, fg_color="transparent")
        v_morph.pack(side="left", padx=5)
        self.morph_label = ctk.CTkLabel(v_morph, text="Kernel: 3x3", font=("Arial", 11))
        self.morph_label.pack()
        self.morph_slider = ctk.CTkSlider(v_morph, from_=3, to=15, number_of_steps=6, command=self.update_morph_label, width=100)
        self.morph_slider.set(3)
        self.morph_slider.pack()
        
        ctk.CTkButton(grp_morph, text="Erode", command=self.apply_erode, height=40, width=70).pack(side="left", padx=5)
        ctk.CTkButton(grp_morph, text="Dilate", command=self.apply_dilate, height=40, width=70).pack(side="left", padx=5)
        ctk.CTkButton(grp_morph, text="Opening", command=self.apply_opening, height=40, width=70).pack(side="left", padx=5)
        ctk.CTkButton(grp_morph, text="Closing", command=self.apply_closing, height=40, width=70).pack(side="left", padx=5)

        # --- SEGMENTATION & FEATURES TAB ---
        grp_thresh = self._create_ribbon_group(tab_seg, "Thresholding")
        v_thresh = ctk.CTkFrame(grp_thresh, fg_color="transparent")
        v_thresh.pack(side="left", padx=5)
        self.thresh_label = ctk.CTkLabel(v_thresh, text="Threshold: 127", font=("Arial", 11))
        self.thresh_label.pack()
        self.thresh_slider = ctk.CTkSlider(v_thresh, from_=0, to=255, number_of_steps=255, command=self.update_thresh_label, width=120)
        self.thresh_slider.set(127)
        self.thresh_slider.pack()
        ctk.CTkButton(grp_thresh, text="Global", command=self.apply_global_threshold, height=40, width=70).pack(side="left", padx=5)
        ctk.CTkButton(grp_thresh, text="Otsu", command=lambda: self.apply_op(segmentation.otsu_threshold), height=40, width=70).pack(side="left", padx=5)
        
        grp_feat = self._create_ribbon_group(tab_seg, "Feature Extraction")
        v_feat = ctk.CTkFrame(grp_feat, fg_color="transparent")
        v_feat.pack(side="left", padx=5)
        self.area_label = ctk.CTkLabel(v_feat, text="Min Area: 100 px", font=("Arial", 11))
        self.area_label.pack()
        self.area_slider = ctk.CTkSlider(v_feat, from_=0, to=5000, number_of_steps=100, command=self.update_area_label, width=120)
        self.area_slider.set(100)
        self.area_slider.pack()
        ctk.CTkButton(grp_feat, text="Extract Objects", command=self.apply_recognition, height=40).pack(side="left", padx=10)

        # ==========================================
        # MAIN WORKSPACE
        # ==========================================
        self.workspace = ctk.CTkFrame(self, fg_color="transparent")
        self.workspace.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.workspace.grid_rowconfigure(0, weight=1)
        self.workspace.grid_columnconfigure(0, weight=3) # Canvas
        self.workspace.grid_columnconfigure(1, weight=1) # Side Panel
        
        # Center: Image Canvas
        self.canvas_frame = ctk.CTkFrame(self.workspace)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.canvas_frame.pack_propagate(False)
        self.img_label = ctk.CTkLabel(self.canvas_frame, text="No Image Loaded", font=("Arial", 16, "bold"), text_color="gray")
        self.img_label.pack(expand=True, fill="both")
        
        # Right Sidebar: Analysis Panel (Histogram)
        self.analysis_panel = ctk.CTkFrame(self.workspace, width=320)
        self.analysis_panel.grid(row=0, column=1, sticky="nsew")
        self.analysis_panel.pack_propagate(False)
        self.analysis_panel.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(self.analysis_panel, text="Image Analysis", font=("Arial", 14, "bold")).pack(pady=15)
        
        self.hist_frame = ctk.CTkFrame(self.analysis_panel)
        self.hist_frame.pack(expand=True, fill="both", padx=10, pady=(0, 10))
        self.hist_frame.pack_propagate(False)
        
        self.fig, self.ax = plt.subplots(figsize=(3.5, 3.5), facecolor='#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#555555')
            
        self.hist_canvas = FigureCanvasTkAgg(self.fig, master=self.hist_frame)
        self.hist_canvas.get_tk_widget().pack(expand=True, fill="both")

    # --- CONTROLLERS (Unchanged functionality) ---
    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if filepath and self.image_state.load(filepath):
            self.refresh_display()
            
    def update_quality_label(self, value):
        self.quality_label.configure(text=f"JPEG Quality: {int(value)}%")

    def save_image(self):
        if self.image_state.current_image is None:
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("Bitmap", "*.bmp")]
        )
        if filepath:
            quality = int(self.quality_slider.get())
            success = compression.save_image(self.image_state.current_image, filepath, quality)
            if success:
                messagebox.showinfo("Success", "Image saved successfully!")
            else:
                messagebox.showerror("Error", "Failed to save the image.")

    def apply_op(self, func, *args):
        if self.image_state.current_image is not None:
            new_img = func(self.image_state.current_image, *args)
            self.image_state.update(new_img)
            self.refresh_display()

    def update_gamma_label(self, value):
        self.gamma_label.configure(text=f"Gamma: {value:.1f}")

    def apply_gamma(self):
        self.apply_op(point_ops.gamma_correction, float(self.gamma_slider.get()))

    def update_blur_label(self, value):
        ksize = int(value)
        if ksize % 2 == 0: ksize += 1
        self.blur_label.configure(text=f"Kernel: {ksize}x{ksize}")

    def apply_mean_blur(self):
        self.apply_op(spatial_filters.mean_blur, int(self.blur_slider.get()))

    def apply_gaussian_blur(self):
        self.apply_op(spatial_filters.gaussian_blur, int(self.blur_slider.get()))

    def update_morph_label(self, value):
        ksize = int(value)
        if ksize % 2 == 0: ksize += 1
        self.morph_label.configure(text=f"Kernel: {ksize}x{ksize}")

    def apply_erode(self):
        self.apply_op(morphology.erode, int(self.morph_slider.get()))

    def apply_dilate(self):
        self.apply_op(morphology.dilate, int(self.morph_slider.get()))
        
    def apply_opening(self):
        self.apply_op(morphology.opening, int(self.morph_slider.get()))
        
    def apply_closing(self):
        self.apply_op(morphology.closing, int(self.morph_slider.get()))

    def update_thresh_label(self, value):
        self.thresh_label.configure(text=f"Threshold: {int(value)}")

    def apply_global_threshold(self):
        self.apply_op(segmentation.global_threshold, int(self.thresh_slider.get()))

    def update_area_label(self, value):
        self.area_label.configure(text=f"Min Area: {int(value)} px")

    def apply_recognition(self):
        self.apply_op(recognition.extract_objects, int(self.area_slider.get()))

    def undo(self):
        if self.image_state.undo():
            self.refresh_display()

    def update_histogram_plot(self):
        self.ax.clear()
        if self.image_state.current_image is not None:
            data = histogram.calculate_histogram(self.image_state.current_image)
            if 'gray' in data:
                self.ax.plot(data['gray'], color='gray')
                self.ax.fill_between(range(256), data['gray'], color='gray', alpha=0.3)
            else:
                self.ax.plot(data['r'], color='#ff5555', label='Red')
                self.ax.plot(data['g'], color='#55ff55', label='Green')
                self.ax.plot(data['b'], color='#5555ff', label='Blue')
            self.ax.set_xlim([0, 256])
            self.ax.set_yticks([]) 
        self.ax.set_facecolor('#2b2b2b')
        self.fig.tight_layout()
        self.hist_canvas.draw()

    def refresh_display(self):
        if self.image_state.current_image is not None:
            pil_img = Image.fromarray(self.image_state.current_image)
            pil_img.thumbnail((900, 600)) 
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            self.img_label.configure(image=ctk_img, text="")
            self.update_histogram_plot()
from fpdf import FPDF

class PDF(FPDF):
    def __init__(self, title='', *args, **kwargs):
        """
        Constructor to allow setting a title dynamically when the PDF is created.
        :param title: Title of the PDF to be added at the top of the first page
        """
        super().__init__(*args, **kwargs)
        self.title = title  # Store the title passed during initialization

    def header(self):
        """
        Add custom title to the top of the first page.
        The title will be centered and in bold.
        """
        if self.title:  # Check if title is set
            self.set_font('Arial', 'B', 12)  # Set font for title (bold, 16 size)
            self.cell(0, 10, self.title, align='C', ln=True)  # Title at top center
            self.ln(8)  # Add space below the title

    def create_table(self, table_data, data_size=10, title_size=12, cell_width=25, x_start='C'):
        """
        This method creates a table based on the provided table_data.
        The table is created below the title (header).
        """
        header = table_data[0]  # Header row
        data = table_data[1:]   # Data rows

        line_height = self.font_size * 2
        col_widths = [cell_width] * len(header)  # Set fixed narrow column width

        # Center table if x_start == 'C'
        if x_start == 'C':
            table_width = sum(col_widths)
            x_start = (self.w - table_width) / 2

        self.set_x(x_start)

        # **Header Processing**
        y_start = self.get_y()
        first_col_x = x_start + col_widths[0]  # X position of vertical separator

        for i, col_title in enumerate(header):
            align = 'C' if i == 0 else 'C'  # **First column header now centered**
            self.set_xy(x_start + sum(col_widths[:i]), y_start)

            if i == 0:
                # **First column header: bold, single-line, CENTERED**
                self.set_font(style='B')
                self.cell(col_widths[i], line_height, col_title, border=0, align=align)
                self.set_font(style='')  # Reset font style
            else:
                # **Other headers: split at spaces into two lines**
                words = col_title.split(" ")
                mid = len(words) // 2
                line1 = " ".join(words[:mid])
                line2 = " ".join(words[mid:])
                self.multi_cell(col_widths[i], line_height / 2, f"{line1}\n{line2}", border=0, align=align)

        self.ln(0)  # Remove extra space

        # **Draw full-height vertical separator**
        y_table_top = y_start
        self.line(first_col_x, y_table_top, first_col_x, self.get_y())

        # **Horizontal line below headers**
        y_after_header = self.get_y()
        self.line(x_start, y_after_header, x_start + sum(col_widths), y_after_header)

        # **Table Data**
        for row in data:
            y_start = self.get_y()
            for i, cell in enumerate(row):
                align = 'R' if i == 0 else 'C'  # First column right-aligned in data, others centered
                self.set_xy(x_start + sum(col_widths[:i]), y_start)
                self.cell(col_widths[i], line_height, str(cell), border=0, align=align)
            self.ln(line_height)

        # **Extend vertical separator to bottom**
        self.line(first_col_x, y_table_top, first_col_x, self.get_y())

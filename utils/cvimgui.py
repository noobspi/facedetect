##########################################
####   OpenCV Python3 Imediate GUI    ####
##########################################
import cv2
import time
import numpy


class GuiError (Exception):
    pass


class Point:
    """
    Represents a Coord for a GUi Widget. 
    if (x,y) is int: coordinates in real pixels (absolut)
    if (x,y) is float: coordinates are normalized: 0.0..1.0
    pivot: defaults to 'sw'. Can be: ('nw', 'ne', 'sw', 'se')
    Raises 'GuiError' if pivot-anchor is unknown.
    """
    def __init__(self, x, y, pivot:str = 'nw') -> None:
        self.x = x
        self.y = y
        self.pivot:str = pivot.lower()

        #pivot_whitelist = ('nw', 'nc', 'ne', 'cw', 'cc', 'ce', 'sw', 'sc', 'se')   #TODO pivot at center is not implemented
        pivot_whitelist = ('nw', 'ne', 'sw', 'se')
        if self.pivot not in pivot_whitelist:
            raise GuiError(f"Pivot-Anchor '{pivot}' unknown. Must be one of {pivot_whitelist}.")


    def get_abs_xy(self, w:int, h:int) -> tuple:
        """
        Returns the Tuple (x:int, y:int) coordinates in absolute pixels.
        w/h: width and height of the canvas/frame in pixels
        """
        xabs:int = self.x if isinstance(self.x, int) else int(self.x * w)
        yabs:int = self.y if isinstance(self.y, int) else int(self.y * h)
        return (xabs, yabs)



class Font:
    """
    Represents a complete OpenCV font definition. See https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html

    fontface: defaults to cv2.FONT_HERSHEY_TRIPLEX
      cv2.FONT_HERSHEY_SIMPLEX          normal size sans-serif font
      cv2.FONT_HERSHEY_PLAIN            small size sans-serif font
      cv2.FONT_HERSHEY_DUPLEX           normal size sans-serif font (more complex than FONT_HERSHEY_SIMPLEX)
      cv2.FONT_HERSHEY_COMPLEX          normal size serif font
      cv2.FONT_HERSHEY_TRIPLEX          normal size serif font (more complex than FONT_HERSHEY_COMPLEX)
      cv2.FONT_HERSHEY_COMPLEX_SMALL    smaller version of FONT_HERSHEY_COMPLEX
      cv2.FONT_HERSHEY_SCRIPT_SIMPLEX   hand-writing style font
      cv2.FONT_HERSHEY_SCRIPT_COMPLEX   more complex variant of FONT_HERSHEY_SCRIPT_SIMPLEX
    fontsize: defaults to 1.2
    fontthikness: defaults to 2
    fontcolor: defaults to (), which means: don't change the text-color, use color from the colorschema instead. 
    """
    #TODO: Use Pillow to render TTF with utf8 strings as image-font?! OpenCVs putText() can only render ASCII.
    # https://stackoverflow.com/questions/37191008/load-truetype-font-to-opencv
    def __init__(self, fontface:int = cv2.FONT_HERSHEY_TRIPLEX, fontsize:float = 1.2, fontthikness:int = 2, fontcolorBGR:tuple = ()) -> None:
        self.fontface:int = fontface
        self.fontsize:float = fontsize
        self.fontthickness:int = fontthikness
        self.fontcolor:tuple = fontcolorBGR


class Gui:
    """
    An simple 'Immediate GUI System' for OpenCV Applications. 100% python, 100% stupid, 100% simple.
    Basic idea taken from Unity3D's Immediate-Mode-GUI-System (See https://docs.unity3d.com/Manual/GUIScriptingGuide.html)
    If you need a complete GUI for your application, then this module might be not the best choice. Use i.e. PySimpleGUI instead for example.
    But if you are only looking for instant 'buttons' and 'labels' for your OpenCV projetct, then you're welcome :)
    """
    COLOR_SCHEMA_BLUE   = 'blue'
    COLOR_SCHEMA_RED    = 'red'
    COLOR_SCHEMA_GREEN  = 'green'
    COLOR_SCHEMA_YELLOW = 'yellow'
    __COLORS = {
        COLOR_SCHEMA_BLUE : {
            'bg': (0xE6, 0xB6, 0x9A),
            'hover': (0xFA, 0xA6, 0x75),
            'line': (0x69, 0x69, 0x69),
            'text': (0x00, 0x00, 0x00),
            'textbg': (0xFF, 0xDE, 0xD2),
            'off': (0xC6, 0x96, 0x6A),
            'on': (0x90, 0xEE, 0x90),
        },
        COLOR_SCHEMA_RED: {
            'bg': (0xAB, 0xB4, 0xE6),
            'hover': (0x89, 0x9b, 0xFC),
            'line': (0x69, 0x69, 0x69),
            'text': (0x00, 0x00, 0x00),
            'textbg': (0xF0, 0xF0, 0xF0),
            'off': (0x8B, 0x94, 0xC6),
            'on': (0x90, 0xEE, 0x90),
        },
        COLOR_SCHEMA_GREEN: {
            'bg': (0xD8, 0xE6, 0xD9),
            'hover': (0xBF, 0xFF, 0xBD),
            'line': (0x69, 0x69, 0x69),
            'text': (0x00, 0x00, 0x00),
            'textbg': (0xF0, 0xF0, 0xF0),
            'off': (0xDA, 0xC6, 0xA9),
            'on': (0x22, 0x80, 0x22),
        },
        COLOR_SCHEMA_YELLOW: {
            'bg': (0xD8, 0xE6, 0xE6),
            'hover': (0xBD, 0xFF, 0xFD),
            'line': (0x69, 0x69, 0x69),
            'text': (0x00, 0x00, 0x00),
            'textbg': (0xF0, 0xF0, 0xF0),
            'off': (0xA8, 0xC6, 0xC6),
            'on': (0x90, 0xEE, 0x90),
        }
    }

    def __init__(self, windowname:str):
        """
        Constructor
        """
        self._window = windowname

        # value-store for persitent ui elements
        self.__store = dict()

        # frame / canvas for painting ui-elements
        self.__frame       = None
        self.__frame_h:int = 0
        self.__frame_w:int = 0

        # mouse event infos
        self.__mouse_x = 0
        self.__mouse_y = 0
        self.__mouse_clicked = False
        self.__mouse_event = 0

        # active/default color schema
        self.__color_schema = self.COLOR_SCHEMA_BLUE


    def _get_color(self, color_name:str):
        """
        Returns the color based on its name (i.e line) and the currently used color-schema 
        """
        if color_name not in self.__COLORS[self.__color_schema]:
            return (0xff, 0xff, 0xff)
        return self.__COLORS[self.__color_schema][color_name]


    def _get_cv2text_size(self, text:str, font:Font=Font(), padding:int=0):
        """
        Returns the real size in pixels of a Text renderd by OpenCV. Return the Tuple (width:int, height:int).
        """ 
        (txtw, txth), txtbaseline = cv2.getTextSize(text, font.fontface, font.fontsize, font.fontthickness)
        w = txtw + 2 * padding
        h = txth + 2 * padding

        return (w, h)


    def set_canvas(self, canvas) -> None:
        """
        Set the canvas (= opencv's numpy array) to paint the gui-elements on (will destroy pixels!)
        """
        self.__frame = canvas
        (self.__frame_h, self.__frame_w) = canvas.shape[:2]


    def set_colorschema(self, schema:str) -> None:
        """
        Set the current color-schema. default=blue
        Raises a 'GuiError', for anknown names!
        """
        if schema in self.__COLORS:
            self.__color_schema = schema
        else:
            raise GuiError("color-schema name not found: {}".format(schema))


    def get_colorschemas(self):
        """
        Returns the list of all available color-schemas
        """
        return list(self.__COLORS.keys())


    def get_value(self, name:str):
        """
        Returns the current value from an UI element by name. 
        Raises a 'GuiError', for unknown names!
        """
        if name in self.__store:
            return self.__store[name]

        raise GuiError("ui-element name not found: {}".format(name))


    def mouse_update(self, event, x, y, flags, param) -> None:
        """
        call this function as (or within) your main mouse eventhandler

        Examample:
            cv2.namedWindow(WINNAME, cv2.WINDOW_AUTOSIZE & cv2.WINDOW_KEEPRATIO)
            gui:cg.Gui = cg.Gui(WINNAME)
            cv2.setMouseCallback(WINNAME, gui.mouse_update)
        """
        # remember mouse coord
        self.__mouse_x = x
        self.__mouse_y = y
        self.__mouse_event = event

        # remember mouse-click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.__mouse_clicked = True
            #print("[DEBUG] UI: mouse-click @ {},{}".format(self._mouse_x, self._mouse_y))
        
        # clear mouse-click, if mouse moved
        if event == cv2.EVENT_MOUSEMOVE:
            self.__mouse_clicked = False




    #####################################
    ###   U I  -  E L E M E N T S     ### 
    #####################################

    def container(self, name:str, x, y, w:int, h:int, bg=True):
        """
        Draws a window/container. optional with caption
        Use this to group ui-elements: they will be drawn relative within this container
        """
        # the very 1st time: init data-store with own coord
        if not name in self.__store:
            self.__store[name] = (x, y, w, h)


        # Draw window
        if bg:
            cv2.rectangle(self.__frame, (x, y), (x + w, y + h), self._get_color('bg'), -1)
            cv2.rectangle(self.__frame, (x, y), (x + w, y + h), self._get_color('line'), 2)
        

    def label(self, caption:str, point:Point, bg:bool=False, font:Font=Font()) -> None:
        """
        Draws a text label

        Params
         caption: the text to draw
         x/y: coord of the label (upper left corner). use an 'int' for pixels and a 'float' between 0.0 and 1.0 for normalized values
         Optional: 'bg' to draw Background and 'size/thickness' for the used font
                    is part of an container: use floats to position element relative within
        """
        #calc real size
        padding = 5
        (x, y) = point.get_abs_xy(self.__frame_w, self.__frame_h)
        (w, h) = self._get_cv2text_size(caption, font, padding)

        # respect pivot-point
        north = True if point.pivot[0] == 'n' else False
        west = True if point.pivot[1] == 'w' else False 
        p1x = x if west else x - w
        p1y = y if north else y - h
        p2x = x if not west else x + w
        p2y = y if not north else y + h

        # Draw the Label
        txtcol = font.fontcolor if font.fontcolor else self._get_color('text')
        # if bg:
        #     cv2.rectangle(self.__frame, (x, y), (x + w, y + h), self._get_color('textbg'), -1)
        # cv2.putText(self.__frame, caption, (x + padding, y + h - padding), font.fontface, font.fontsize, txtcol, font.fontthickness)
        if bg:
            cv2.rectangle(self.__frame, (p1x, p1y), (p2x, p2y), self._get_color('textbg'), -1)
        cv2.putText(self.__frame, caption, (p1x + padding, p2y - padding), font.fontface, font.fontsize, txtcol, font.fontthickness)
        

    def fpscounter(self, point:Point, update_interval:float = 1.0, font:Font=Font()) -> None:
        """
        Draws a FPS-Counter.
        update_interval in seconds.
        """
        store_name = '_fpscnt'
        fps_now = time.time()
        fps_start = fps_now
        fps_cnt = 0
        fps_avg = 1
        
        # the very 1st time: init data-store, else read fps-data from store
        if  store_name not in self.__store:
            self.__store[store_name] = (fps_now, 0, fps_avg)
        else:
            (fps_start, fps_cnt, fps_avg) = self.__store[store_name]

        # if N seconds passed calc fps-avg and reset store for next measurement. Or count the passed frame
        if fps_start + update_interval < fps_now:
            fps_avg = int(fps_cnt / update_interval)
            self.__store[store_name] = (fps_now, 0, fps_avg)
        else:
            self.__store[store_name] = (fps_start, fps_cnt+1, fps_avg)
        
        # draw FPS-Counter as a label
        self.label(f"{fps_avg}fps", point, bg=True, font=font)


    def button(self, caption:str, point:Point, w:int=1, h:int=1, font:Font=Font()) -> bool:
        """
        Draws a Button. 
        Returns True, if clicked (= mouse-down, while hovering over the button)

        Params
         caption: the text on the button
         startx/starty: coord of the label (upper left corner). use an 'int' for pixels and a 'float' between 0.0 and 1.0 for normalized values
         Optional: width and height of the button. default and fallback: minimal size for the caption-text
                   is part of an container: use floats to position element relative within
        """
        # calc real size
        padding = 10
        line_width = 2
        (x, y) = point.get_abs_xy(self.__frame_w, self.__frame_h)
        (txtw, txth) = self._get_cv2text_size(caption, font)
        w = max(txtw + 2 * padding, w)
        h = max(txth + 2 * padding, h)

        # respect pivot-point
        north = True if point.pivot[0] == 'n' else False
        west = True if point.pivot[1] == 'w' else False 
        p1x = x if west else x - w
        p1y = y if north else y - h
        p2x = x if not west else x + w
        p2y = y if not north else y + h

        # check if mouse is over the button
        mouse_over = ((self.__mouse_x >= p1x and self.__mouse_x <= p2x) and
                      (self.__mouse_y >= p1y and self.__mouse_y <= p2y))
        
        # set button color: bg as default, hover when mouse is over it
        btn_col = self._get_color('hover') if mouse_over else self._get_color('bg')

        # Draw Button
        cv2.rectangle(self.__frame, (p1x, p1y), (p2x, p2y), btn_col, -1)
        cv2.rectangle(self.__frame, (p1x, p1y), (p2x, p2y), self._get_color('line'), line_width)
        cv2.putText(self.__frame, caption, (p1x + padding, p2y - int((h - txth) / 2)), font.fontface, font.fontsize, self._get_color('text'), font.fontthickness)
        
        # react on mouse-click
        if self.__mouse_clicked and mouse_over:
            # consume mouse-click
            self.__mouse_clicked = False
            return True
        
        # button is not clicked by the user
        return False


    def checkbox(self, name:str, caption:str, point:Point, checked:bool=False, bg:bool=True, w:int=1, h:int=1, font:Font=Font()) -> bool:
        """
        Draws a Checkbox & Returns True, when clicked/toogled (mouse-down over the checkbox).

        Params
         caption: the text for the checkbox (alligned on the right)
         x/x: coord of the label (upper left corner). use an 'int' for pixels and a 'float' between 0.0 and 1.0 for normalized values
         Optional: 
          width and height of the checkbox. default and fallback: minimal size for the caption-text
          is part of an container: use floats to position element relative within
          checked/value, background
        """
        chkbox_has_toogled = False       
        # the very 1st time: init data-store with default value
        if not name in self.__store:
            self.__store[name] = checked

        # calc real size
        padding_outside = 5
        padding_inside  = 15    # padding betwenn green-light and text
        (x, y) = point.get_abs_xy(self.__frame_w, self.__frame_h)
        (txtw, txth) = self._get_cv2text_size(caption, font)
        txw = txtw + 2 * padding_outside
        txh = txth + 2 * padding_outside
        cbw = txth
        cbh = txth
        w = cbw + padding_inside + txw
        h = txh

        # respect pivot-point
        north = True if point.pivot[0] == 'n' else False
        west = True if point.pivot[1] == 'w' else False 
        p1x = x if west else x - w
        p1y = y if north else y - h
        p2x = x if not west else x + w
        p2y = y if not north else y + h

        # check if mouse is over the button
        mouse_over = ((self.__mouse_x >= p1x and self.__mouse_x <= p2x) and
                      (self.__mouse_y >= p1y and self.__mouse_y <= p2y))
        
        # react on mouse-click (toogle the value)
        last_value = self.__store[name]
        if self.__mouse_clicked and mouse_over:
            self.__store[name] = not last_value
            chkbox_has_toogled = True
            # consume mouse-click
            self.__mouse_clicked = False


        # Draw the Checkbox
        if bg:
            # set button color: bg as default, hover when mouse is over it
            cb_col = self._get_color('hover') if mouse_over else self._get_color('bg')
            cv2.rectangle(self.__frame, (p1x, p1y), (p2x, p2y), cb_col, -1)
        if self.__store[name]:
            cv2.rectangle(self.__frame, (p1x + padding_outside, p1y + padding_outside), (p1x + padding_outside + cbw, p1y + padding_outside + cbh), self._get_color('on'), -1)
        else:
            cv2.rectangle(self.__frame, (p1x + padding_outside, p1y + padding_outside), (p1x + padding_outside + cbw, p1y + padding_outside + cbh), self._get_color('off'), -1)        
        cv2.rectangle(self.__frame, (p1x + padding_outside, p1y + padding_outside), (p1x + padding_outside + cbw, p1y + padding_outside + cbh), self._get_color('line'), )
        cv2.putText(self.__frame, caption, (p1x + padding_outside + cbw + padding_inside, p2y - padding_outside), font.fontface, font.fontsize, self._get_color('text'), font.fontthickness)
        
        return chkbox_has_toogled

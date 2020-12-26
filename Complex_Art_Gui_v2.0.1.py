import tkinter as tk
import os
import primes_list
import numpy as np
import matplotlib.pyplot as plt
import random as rd


class Draw_Complex:
    def generate_5_complex(self, value_1, value_2, value_3, value_4,value_5, 
                           t_range=2000, t_spread=1000, 
                           color_1='black',color_2='cyan', color_3='blue', color_4='red',color_5='yellow', 
                           size_1=60, size_2=40,size_3=25,size_4=15, size_5=5, 
                           alpha_1=0.25,alpha_2=0.25, alpha_3=0.25, alpha_4=0.25,alpha_5=0.25, 
                           offset_1=1.45, offset_2=1.49,offset_3=1.505,offset_4=1.535, offset_5=1.57, 
                           enable_1=109751, enable_2=139891,enable_3=65063,enable_4=30059, enable_5=209311):
        self.hot_ls=[]
        for i in range(12):
            self.hot_ls+=[[]]
        (eq_0_r, eq_1_r, eq_2_r, eq_3_r, eq_4_r, eq_5_r, 
            eq_0_i, eq_1_i, eq_2_i, eq_3_i, eq_4_i, eq_5_i) = self.hot_ls
        for i in range(t_range):
            eq_r1, eq_r2, eq_r3, eq_r4, eq_r5 = ((np.real(np.e**(1j*2*np.pi*(i/t_spread)*value_1))*enable_1),
                                                 (np.real(np.e**(1j*2*np.pi*(i/t_spread)*value_2))*enable_2),
                                                 (np.real(np.e**(1j*2*np.pi*(i/t_spread)*value_3))*enable_3),
                                                 (np.real(np.e**(1j*2*np.pi*(i/t_spread)*value_4))*enable_4),
                                                 (np.real(np.e**(1j*2*np.pi*(i/t_spread)*value_5))*enable_5))
            eq_0_r += [eq_r1+eq_r2+eq_r3+eq_r4+eq_r5]
            eq_i1, eq_i2, eq_i3, eq_i4, eq_i5 = ((np.imag(np.e**(1j*2*np.pi*(i/t_spread)*value_1))*enable_1),
                                                 (np.imag(np.e**(1j*2*np.pi*(i/t_spread)*value_2))*enable_2),
                                                 (np.imag(np.e**(1j*2*np.pi*(i/t_spread)*value_3))*enable_3),
                                                 (np.imag(np.e**(1j*2*np.pi*(i/t_spread)*value_4))*enable_4),
                                                 (np.imag(np.e**(1j*2*np.pi*(i/t_spread)*value_5))*enable_5))
            eq_0_i += [eq_i1+eq_i2+eq_i3+eq_i4+eq_i5]
        for i in range(int((len(eq_0_i)+len(eq_0_r))/2)):
            eq_1_r += [(eq_0_r[i])/(offset_1)]
            eq_1_i += [(eq_0_i[i])/(offset_1)]
            eq_2_r += [(eq_0_r[i])/(offset_2)]
            eq_2_i += [(eq_0_i[i])/(offset_2)]
            eq_3_r += [(eq_0_r[i])/(offset_3)]
            eq_3_i += [(eq_0_i[i])/(offset_3)]
            eq_4_r += [(eq_0_r[i])/(offset_4)]
            eq_4_i += [(eq_0_i[i])/(offset_4)]
            eq_5_r += [(eq_0_r[i])/(offset_5)]
            eq_5_i += [(eq_0_i[i])/(offset_5)]
        img_lbl = [pot.index(value_1), pot.index(value_2), pot.index(value_3), pot.index(value_4),pot.index(value_5)]
        with plt.style.context('seaborn-dark'):
            plt.scatter(eq_1_r,eq_1_i,s=size_1,alpha=alpha_1,c=color_1)
            plt.scatter(eq_2_r,eq_2_i,s=size_2,alpha=alpha_2,c=color_2)
            plt.scatter(eq_3_r,eq_3_i,s=size_3,alpha=alpha_3,c=color_3)
            plt.scatter(eq_4_r,eq_4_i,s=size_4,alpha=alpha_4,c=color_4)
            plt.scatter(eq_5_r,eq_5_i,s=size_5,alpha=alpha_5,c=color_5)
            plt.xlabel(str(img_lbl), fontsize = 18)
            plt.xticks([])
            plt.yticks([])
            plt.savefig('plotimg/imgraphguitest', dpi=120)
            plt.close()

class d_f_T:
    def dft_calc(self, input_array):
        self.dft_dict = {}
        self.dft_dict['frequency_var'] = []
        self.dft_dict['amplitude_var'] = []
        self.dft_dict['phase_var'] = []
        self.dft_dict['coord_var'] = []
        self.x_real = 0
        self.y_imag = 0
        self.range_var = len(input_array)
        for k in range(self.range_var):  
            for n in range(self.range_var):
                angle_var = (((2*np.pi)*k*n)/self.range_var)
                self.x_real += ((input_array[n])* np.cos(angle_var))
                self.y_imag += (-(input_array[n])* np.sin(angle_var))
            self.x_real = self.x_real / self.range_var
            self.y_imag = self.y_imag / self.range_var
            self.dft_dict['frequency_var'] += [k]
            self.dft_dict['amplitude_var'] += [((self.x_real**2)+(self.y_imag**2))**(1/2)]
            self.dft_dict['phase_var'] += [np.arctan2(self.y_imag, self.x_real)]
            self.dft_dict['coord_var'] += [(self.x_real,self.y_imag)]
        return(self.dft_dict)
    def plot_phase_amp(self, s_list):
        a_ls,b_ls,c_ls,d_ls,e_ls=[],[],[],[],[]
        for i in s_list:
            a_ls += [int(i[0])]
            b_ls += [int(i[1])]
            c_ls += [int(i[2])]
            d_ls += [int(i[3])]
            e_ls += [int(i[4])]
        a_ls_ft = self.dft_calc(a_ls)
        b_ls_ft = self.dft_calc(b_ls)
        c_ls_ft = self.dft_calc(c_ls)
        d_ls_ft = self.dft_calc(d_ls)
        e_ls_ft = self.dft_calc(e_ls)
        result_tpl=((round(sum(e_ls_ft['amplitude_var'][1::])/len(e_ls_ft['frequency_var'][1::]),2),
                    round(sum(e_ls)/len(e_ls),2)),
                    (round(sum(b_ls_ft['amplitude_var'][1::])/len(b_ls_ft['frequency_var'][1::]),2),
                    round(sum(b_ls)/len(b_ls),2)),
                    (round(sum(c_ls_ft['amplitude_var'][1::])/len(c_ls_ft['frequency_var'][1::]),2),
                    round(sum(c_ls)/len(c_ls),2)),
                    (round(sum(d_ls_ft['amplitude_var'][1::])/len(d_ls_ft['frequency_var'][1::]),2),
                    round(sum(d_ls)/len(d_ls),2)),
                    (round(sum(e_ls_ft['amplitude_var'][1::])/len(e_ls_ft['frequency_var'][1::]),2),
                    round(sum(e_ls)/len(e_ls),2)))
        with plt.style.context('seaborn-dark'):
            plt.plot(a_ls_ft['frequency_var'][1::],a_ls_ft['amplitude_var'][1::], color='black', label='A')
            plt.plot(b_ls_ft['frequency_var'][1::],b_ls_ft['amplitude_var'][1::], color='cyan', label='B')
            plt.plot(c_ls_ft['frequency_var'][1::],c_ls_ft['amplitude_var'][1::], color='blue', label='C')
            plt.plot(d_ls_ft['frequency_var'][1::],d_ls_ft['amplitude_var'][1::], color='red', label='D')
            plt.plot(e_ls_ft['frequency_var'][1::],e_ls_ft['amplitude_var'][1::], color='yellow', label='E')
            plt.xlabel(str(result_tpl), fontsize = 10)
            plt.grid()
            plt.legend(loc=9)
            plt.savefig('plotimg/imgraphguitest', dpi=120)
            plt.close()  


class Handle_File(Draw_Complex, d_f_T):
    def ls_directory(self, directory='plotimg', filetype='.png'):
        ls_dir_raw = os.listdir(directory)
        ls_dir = []
        for i in ls_dir_raw:
            if filetype == 'any':
                ls_dir += [i]
            elif filetype in i:
                ls_dir += [i]
            else:
                pass
        return ls_dir
    def make_file(self, f_name='eqvar.txt', f_content=''):
        save_txt = open('{}'.format(f_name), 'w')
        save_txt.write(f_content)
        save_txt.close()
    def open_file(self, f_name='eqvar.txt'):
        raw_txt = open('{}'.format(f_name), 'r')
        str_read = raw_txt.read()
        raw_txt.close()
        return str_read
    def read_list_txt(self, f_name='eqvar.txt'):
        txt_ls = self.open_file(f_name).split('\n')[1::]
        new_ls = []
        for i in txt_ls:
            try:
                hot_var = eval(i)
                if hot_var not in new_ls:
                    new_ls += [hot_var]
            except:
                pass
        return new_ls
    def append_file(self, append_str, f_name='eqvar.txt'):
        str_read = self.open_file(f_name)
        str_read = str_read + '\n' + append_str
        save_txt = open('{}'.format(f_name), 'w')
        save_txt.write(str_read)
        save_txt.close()
    def append_list(self, append_ls, f_name='eqvar.txt'):
        ls_read = self.read_list_txt(f_name)
        for i in append_ls:
            if i not in ls_read:
                ls_read += [i]
        self.make_file(f_name)
        for i in ls_read:
            self.append_file(str(i), f_name)
    def append_entry(self, a1_int, b2_int, c3_int, d4_int, e5_int, f_name='eqvar.txt'):
        if type(a1_int) == str:
            entry_var = [int(a1_int),int(b2_int),int(c3_int),int(d4_int),int(e5_int)]
        elif type(a1_int) == int:
            entry_var = [a1_int, b2_int, c3_int, d4_int, e5_int]
        txt_ls = [entry_var]
        self.append_list(txt_ls, f_name)
    def draw_list(self, s_list):
        for i in s_list:
            value_1 = pot[int(i[0])]
            value_2 = pot[int(i[1])]
            value_3 = pot[int(i[2])]
            value_4 = pot[int(i[3])]
            value_5 = pot[int(i[4])]
            self.generate_5_complex(value_1, value_2, value_3, value_4, value_5)
    def draw_entry(self, a1_int, b2_int, c3_int, d4_int, e5_int):
        value_1 = pot[int(a1_int)]
        value_2 = pot[int(b2_int)]
        value_3 = pot[int(c3_int)]
        value_4 = pot[int(d4_int)]
        value_5 = pot[int(e5_int)]
        self.generate_5_complex(value_1, value_2, value_3, value_4, value_5)
    def get_sample_ls(self, sample_length):
        sample_ls=[]
        for i in range(sample_length):#  .0
            a_var = pot[int(((rd.randrange(13,31))*np.pi-(i/1))/1.0+(i/1))]
            b_var = pot[int(((rd.randrange(13,31))*np.pi-(i/1))/1.0+(i/1))]
            c_var = pot[int(((rd.randrange(13,31))*np.pi-(i/1))/1.0+(i/1))]
            d_var = pot[int(((rd.randrange(13,31))*np.pi-(i/1))/1.0+(i/1))]
            e_var = pot[int(((rd.randrange(13,31))*np.pi-(i/1))/1.0+(i/1))]
            sample_ls+=[[pot.index(a_var),pot.index(b_var),pot.index(c_var),pot.index(d_var),pot.index(e_var)]]
        return sample_ls

class Menu_Method:
    def add_menubutton(self):
        self.setting.set('Select')
        self.setting_var = tk.Label(self, textvariable=self.setting)
        self.menubutton = tk.Menubutton(text="Options", border=2)
        self.menu = tk.Menu(self.menubutton, tearoff=0,  border=2)
        self.menu.add_command(label='Draw', command=self.on_menu_draw_clicked)
        self.menu.add_command(label='Find', command=self.on_menu_find_clicked)
        self.menu.add_command(label='Load', command=self.on_menu_load_clicked)
        self.menu.add_command(label='Spam', command=self.on_menu_spam_clicked)
        self.menu.add_command(label='Return', command=self.on_menu_return_clicked)
        self.menubutton['menu'] = self.menu
        self.calc_btn = tk.Button(self, text="Confirm Selection", command=self.on_menu_select_clicked)
        self.menubutton.tk_setPalette(background='silver')
        self.setting_var.grid(column=0,row=0)
        self.menubutton.grid(column=1,row=0)
        self.calc_btn.grid(column=2,row=0)
    def on_menu_draw_clicked(self):
        self.setting.set('Draw')
    def on_menu_find_clicked(self):
        self.setting.set('Find')
    def on_menu_load_clicked(self):
        self.setting.set('Sort')
    def on_menu_spam_clicked(self):
        self.setting.set('Spam')
    def on_menu_return_clicked(self):
        self.setting.set('Return')
    def on_menu_select_clicked(self):
        self.hot_str = self.setting.get()
        if self.hot_str == 'Draw':
            self.current_instance = 'Draw'
            self.draw_image()
        elif self.hot_str == 'Find':
            self.current_instance = 'Find'
            self.i_index=0
            self.render_doc()
        elif self.hot_str == 'Spam':
            self.current_instance = 'Spam'
            self.i_index=0
            self.hot_str = self.get_sample_ls(10) 
            self.render_spam() 
        elif self.hot_str == 'Load':
            self.current_instance = 'Load'
            self.render_load()
        elif self.hot_str == 'Return':
            self.current_instance = 'Return'
            self.init_main()

class Draw_Method:
    def draw_image(self):
        self.remove_init()
        self.title("CAM_v1: Draw from 5 integer indices")
        self.dot_value1.set(self.d_value1)
        self.dot_value2.set(self.d_value2)
        self.dot_value3.set(self.d_value3)
        self.dot_value4.set(self.d_value4)
        self.dot_value5.set(self.d_value5)
        self.dot_colour1_label = tk.Label(self, text="Dot 1")
        self.dot_value1_entry = tk.Entry(self, width=5, textvariable=self.dot_value1)
        self.dot_colour2_label = tk.Label(self, text="Dot 2")
        self.dot_value2_entry = tk.Entry(self, width=5, textvariable=self.dot_value2)
        self.dot_colour3_label = tk.Label(self, text="Dot 3")
        self.dot_value3_entry = tk.Entry(self, width=5, textvariable=self.dot_value3)
        self.dot_colour4_label = tk.Label(self, text="Dot 4")
        self.dot_value4_entry = tk.Entry(self, width=5, textvariable=self.dot_value4)
        self.dot_colour5_label = tk.Label(self, text="Dot 5")
        self.dot_value5_entry = tk.Entry(self, width=5, textvariable=self.dot_value5)
        self.tk_frame = tk.Canvas(self, width=self.frame_width, height=self.frame_height)
        self.draw_btn = tk.Button(self, text="draw", command=self.get_image)
        self.draw_btn.grid(column=1,row=5)
        self.dot_colour1_label.grid(column=1,row=3)
        self.dot_value1_entry.grid(column=1,row=4)
        self.dot_colour2_label.grid(column=2,row=3)
        self.dot_value2_entry.grid(column=2,row=4)
        self.dot_colour3_label.grid(column=3,row=3)
        self.dot_value3_entry.grid(column=3,row=4)
        self.dot_colour4_label.grid(column=4,row=3)
        self.dot_value4_entry.grid(column=4,row=4)
        self.dot_colour5_label.grid(column=5,row=3)
        self.dot_value5_entry.grid(column=5,row=4)
        self.add_menubutton()
        self.eval('tk::PlaceWindow %s center' % self.winfo_pathname(self.winfo_id()))
    def get_image(self):
        self.title("CAM_v1: Displaying drawn image")
        self.d_value1 = self.dot_value1.get()
        self.d_value2 = self.dot_value2.get()
        self.d_value3 = self.dot_value3.get()
        self.d_value4 = self.dot_value4.get()
        self.d_value5 = self.dot_value5.get()
        self.remove_init()
        self.tk_frame = tk.Canvas(self, width=self.frame_width, height=self.frame_height) 
        self.return_btn = tk.Button(self, text="return", command=self.draw_image)
        self.save_btn = tk.Button(self, text="save", command=self.save_drawing)
        self.generate_5_complex(pot[eval(self.d_value1)], pot[eval(self.d_value2)], pot[eval(self.d_value3)], 
                                pot[eval(self.d_value4)], pot[eval(self.d_value5)]) 
        self.fetching_img()
        self.tk_frame.create_image(300, 250, image=self.img_var)
        self.tk_frame.grid(column=2,row=2, columnspan=3)
        self.return_btn.grid(column=3,row=0)
        self.save_btn.grid(column=3,row=1)
        self.add_menubutton()
        self.eval('tk::PlaceWindow %s center' % self.winfo_pathname(self.winfo_id()))
    def save_drawing(self):
        raw_ls = self.read_list_txt()
        self.make_file(self.id_txt_f)
        raw_ls += [[eval(self.d_value1), eval(self.d_value2), eval(self.d_value3), eval(self.d_value4), eval(self.d_value5)]]
        self.append_list(raw_ls) 

class Render_List:
    def get_image_row(self):
        self.tk_frame = tk.Canvas(self, width=self.frame_width, height=self.frame_height)    
        self.image_max_var = tk.Label(self, text=self.m_index)
        self.image_nr_var = tk.Label(self, text=self.i_index)
        self.generate_5_complex(pot[int(self.hot_str[int(self.i_index)][0])],
                                pot[int(self.hot_str[int(self.i_index)][1])],
                                pot[int(self.hot_str[int(self.i_index)][2])],
                                pot[int(self.hot_str[int(self.i_index)][3])],
                                pot[int(self.hot_str[int(self.i_index)][4])]) 
        self.fetching_img()  
        self.tk_frame.create_image(300, 250, image=self.img_var)
        self.tk_frame.grid(column=2,row=3, columnspan=3)
        self.image_nr_var.grid(column=1,row=1)
        self.image_max_var.grid(column=2,row=1)

class Display_File(Render_List):
    def render_doc(self):
        self.remove_init()
        self.title("CAM_v1: Render saved rrawings")
        self.hot_str = self.read_list_txt(self.id_txt_f)
        self.m_index = len(self.hot_str)-1
        self.get_image_row()
        self.render_images_s()
        self.add_menubutton()
        self.eval('tk::PlaceWindow %s center' % self.winfo_pathname(self.winfo_id()))
    def render_images_s(self):
        self.next_btn = tk.Button(self, text="Next", command=self.render_next)
        self.next_btn.grid(column=3,row=0)
        self.throw_btn = tk.Button(self, text="Throw", command=self.render_throw)
        self.throw_btn.grid(column=4,row=0)
        self.back_btn = tk.Button(self, text="Back", command=self.render_back)
        self.back_btn.grid(column=3,row=1)
        self.load_btn = tk.Button(self, text="DFT", command=self.render_open)
        self.load_btn.grid(column=4,row=1)
    def render_open(self, *args):
        self.render_load()
    def render_next(self, *args):
        if self.i_index < self.m_index:
            self.i_index += 1
            self.render_doc()
    def render_back(self, *args):
        if self.i_index > 0:
            self.i_index -= 1
            self.render_doc()
    def render_throw(self, *args):
        self.hot_str.remove(self.hot_str[self.i_index])
        if self.i_index != 0:
            self.i_index-=1
            
        self.make_file(self.id_txt_f)
        self.append_list(self.hot_str, self.id_txt_f)
        self.render_doc()

class Generate_Images(Render_List):
    def render_spam(self):
        self.remove_init()
        self.title("CAM_v1: filter through set config")
        self.m_index = len(self.hot_str)-1
        self.get_image_row()
        self.render_spam_s()
        self.add_menubutton()
        self.eval('tk::PlaceWindow %s center' % self.winfo_pathname(self.winfo_id()))
    def render_spam_s(self):
        self.next_btn = tk.Button(self, text="Next", command=self.s_render_next)
        self.next_btn.grid(column=3,row=0)
        self.save_btn = tk.Button(self, text="Save", command=self.render_save)
        self.save_btn.grid(column=4,row=0)
        self.back_btn = tk.Button(self, text="Back", command=self.s_render_back)
        self.back_btn.grid(column=3,row=1)
        self.new_btn = tk.Button(self, text="New", command=self.s_render_new)
        self.new_btn.grid(column=4,row=1)
    def s_render_new(self, *args):
        self.hot_str = self.get_sample_ls(10) 
        self.i_index = 0
        self.render_spam()
    def s_render_next(self, *args):
        if self.i_index < self.m_index:
            self.i_index += 1
            self.render_spam()
    def s_render_back(self, *args):
        if self.i_index > 0:
            self.i_index -= 1
            self.render_spam()
    def render_save(self, *args):
        self.s_ls=self.read_list_txt()
        self.s_ls+=[self.hot_str[self.i_index]]
        self.make_file('eqvar.txt')
        self.append_list(self.s_ls, 'eqvar.txt')   
        self.render_spam()

class Load_Method:
    def render_load(self):
        self.remove_init()
        self.title("Complex Art Maker v1 Menu")
        self.tk_frame = tk.Canvas(self, width=self.frame_width, height=self.frame_height)  
        self.hot_str = self.read_list_txt(self.id_txt_f)
        self.plot_phase_amp(self.hot_str)
        self.fetching_img()  
        self.tk_frame.create_image(350, 300, image=self.img_var)
        self.tk_frame.grid(column=1,row=1, columnspan=3)
        self.add_menubutton()
        self.eval('tk::PlaceWindow %s center' % self.winfo_pathname(self.winfo_id()))

class Index_Method:
    def init_main(self):
        self.remove_init()
        self.title("Complex Art Maker v1 Menu")
        self.add_menubutton()
        self.eval('tk::PlaceWindow %s center' % self.winfo_pathname(self.winfo_id()))

class Tk_Main(tk.Tk, Index_Method, Draw_Method, Display_File, Handle_File, 
              Menu_Method, Load_Method, Generate_Images):
    frame_width = 800
    frame_height = 600
    i_index = 0
    current_instance = 'Return'
    id_txt_f = 'misc_all_list.txt'
    d_value1 = '1'
    d_value2 = '2'
    d_value3 = '3'
    d_value4 = '2'
    d_value5 = '1'
    c_setting = ''
    objects_g = ['tk.Canvas.destroy(self.tk_frame)',
                 'tk.Menubutton.destroy(self.menubutton)',
                 'tk.Menu.destroy(self.menu)',
                 'tk.Label.destroy(self.setting_var)',
                 'tk.Label.destroy(self.image_nr_var)',
                 'tk.Label.destroy(self.image_max_var)',
                 'tk.Label.destroy(self.dot_colour1_label)',
                 'tk.Label.destroy(self.dot_colour2_label)',
                 'tk.Label.destroy(self.dot_colour3_label)',
                 'tk.Label.destroy(self.dot_colour4_label)',
                 'tk.Label.destroy(self.dot_colour5_label)',
                 'tk.Button.destroy(self.calc_btn)',
                 'tk.Button.destroy(self.return_btn)',
                 'tk.Button.destroy(self.next_btn)',
                 'tk.Button.destroy(self.load_btn)',
                 'tk.Button.destroy(self.throw_btn)',
                 'tk.Button.destroy(self.back_btn)',
                 'tk.Button.destroy(self.save_btn)',
                 'tk.Button.destroy(self.draw_btn)',
                 'tk.Button.destroy(self.new_btn)',
                 'tk.Entry.destroy(self.dot_value1_entry)',
                 'tk.Entry.destroy(self.dot_value2_entry)',
                 'tk.Entry.destroy(self.dot_value3_entry)',
                 'tk.Entry.destroy(self.dot_value4_entry)',
                 'tk.Entry.destroy(self.dot_value5_entry)'] 
    def __init__(self):
        super().__init__()
        self.geometry("{}x{}".format(int(self.frame_width),int(self.frame_height)))
        self.bind("<Key>", self.key_pressed)
        self.dot_value1 = tk.StringVar()
        self.dot_value2 = tk.StringVar()
        self.dot_value3 = tk.StringVar()
        self.dot_value4 = tk.StringVar()
        self.dot_value5 = tk.StringVar()
        self.setting = tk.StringVar()
        self.image_nr = tk.StringVar()
        self.doc_entry = tk.StringVar()
        self.init_main()  
    def remove_init(self):
        for i in self.objects_g:
            try:
                eval(i)
            except:
                pass
    def key_pressed(self, tk_command):
        #self.tk_frame = tk_command.widget.tk_frame()
        if self.current_instance == 'Spam':
            if tk_command.char == 'q':
                self.s_render_back()
            elif tk_command.char == 's':
                self.render_save()
            elif tk_command.char == 'w':
                self.s_render_new()
            elif tk_command.char == 'e':
                self.s_render_next()
        elif self.current_instance == 'Find':
            if tk_command.char == 'q':
                self.render_back()
            elif tk_command.char == 'w':
                self.render_throw()
            elif tk_command.char == 'e':
                self.render_next()
            elif tk_command.char == 's':
                self.render_open()

    def resource_path(self, relative_path):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)
    def fetching_img(self):
        self.img_var = tk.PhotoImage(file=self.resource_path("plotimg\imgraphguitest.png"))


pot = primes_list.primesofthousands
init_main = Tk_Main()
init_main.mainloop()
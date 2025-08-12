from WF_SDK import device, scope, wavegen, tools, error   # import instruments
import numpy as np
import matplotlib.pyplot as plt   # needed for plotting
import time
from scipy import signal
from scipy.signal import correlate
from Decorators import cronometer


class AnalogDiscovery:
    def __init__(self, sampling_frequency = 50000000, amplitude_range = 50, chan1Probe = 10, chan2Probe = 1):
        """
        Init the analog discovery instrument:
            -DO NOT USE IF YOU DONT KNOW WHAT YOU ARE DOING!!, ask Carlos for help and keep the device working please :)
        Args:
            , 
        """
        self.device_data = device.open()
        print("Opening Device")
        self.ConfigureScope(sampling_frequency=sampling_frequency, amplitude_range=amplitude_range)
        self.sampling_frequency = sampling_frequency
        self.amplitude_range = amplitude_range
        self.C1probe = chan1Probe
        self.C2probe = chan2Probe
        if self.C1probe != 1 or self.C2probe != 1:
            self.probes = True
        else:
            self.probes = False
        
    def ConfigureScope(self, sampling_frequency: float = 20000000, buffer_size: int = 0, offset: int = 0, amplitude_range: int = 5):
        """
        Configures the Osciloscope with default trigger options.
            Args:
            - self.sampling_frequency: device sampling frec (Goes over 30MHz)
            - self.amplitude_range: Range of amplitudes going in the scope
        """
        scope.open(self.device_data, sampling_frequency = sampling_frequency, amplitude_range= amplitude_range, amplitude_range2 = 0.05, offset = offset, buffer_size = buffer_size)
        scope.trigger(self.device_data, enable=True, source=scope.trigger_source.analog, channel=1, level=0)        
        time.sleep(1.5) #Sleeps to let the fpga configure.
        print("AD Osciloscope Configured")
        
    def ChangeTrigger(self, chanel = 1, level = 0):
        """
        set up triggering

        parameters: - device data

        enable / disable triggering with True/False
        trigger source - possible: none, analog, digital, external[1-4]
        trigger channel - possible options: 1-4 for analog, or 0-15 for digital
        auto trigger timeout in seconds, default is 0
        trigger edge rising - True means rising, False means falling, default is rising
        trigger level in Volts, default is 0V
        """
        scope.trigger(self.device_data, enable=True, source=scope.trigger_source.analog, channel=chanel, level=level)
        time.sleep(1.5)

    def AddProbe(self, chan1Probe, chan2Probe):
        self.probes = True
        self.C1probe = chan1Probe
        self.C2probe = chan2Probe

    def measureBuffers(self):
        #print("Measuring Buffers...")
        self.buffer1, self.buffer2 = scope.record_double_corrected(self.device_data)
        if self.probes:
            self.buffer1 = np.asarray(self.buffer1,dtype=float)*self.C1probe
            self.buffer2 = np.asarray(self.buffer2,dtype=float)*self.C2probe
    
    def calculate_phase(self, _print=False):
        """
        Calcula la diferencia de fase entre dos señales con debug detallado.
        Réplica corregida de la función JavaScript original.
        
        Parameters:
        - debug: mostrar información de debug
        
        Returns:
        - fase: diferencia de fase en grados (-180 a +180)
        """
        # Convertir a numpy arrays
        self.buffer1 = np.array(self.buffer1, dtype=np.float64)
        self.buffer2 = np.array(self.buffer2, dtype=np.float64)
        cnt = len(self.buffer1)
        
        # Remove DC offset
        avg1 = np.mean(self.buffer1)
        avg2 = np.mean(self.buffer2)
        
        buffer1_clean = self.buffer1 - avg1
        self.buffer2_clean = self.buffer2 - avg2
        
        # Compute unsigned phase usando producto punto
        sum1 = np.sum(buffer1_clean * buffer1_clean)
        sum2 = np.sum(self.buffer2_clean *self.buffer2_clean)
        sum12 = np.sum(buffer1_clean *self.buffer2_clean)
        
        # Evitar división por cero y valores fuera del rango de arccos
        denominator = np.sqrt(sum1 * sum2)
        if denominator == 0:
            print("Error: denominador cero")
            self.fase = 0
        
        cos_phase = sum12 / denominator
        
        # Clampear para evitar errores numéricos en arccos
        cos_phase = np.clip(cos_phase, -1.0, 1.0)
        
        phase_deg = np.arccos(cos_phase) * 180 / np.pi
        
        if _print:
            print(f"   sum1: {sum1:.2e}, sum2: {sum2:.2e}, sum12: {sum12:.2e}")
            print(f"   cos_phase: {cos_phase:.4f}")
            print(f"   Fase sin signo: {phase_deg:.2f}°")
        
        # Estimate lag sign with correlation - MÉTODO CORREGIDO
        max_lag = min(200, cnt//4)  # Limitar para evitar problemas con señales cortas
        
        # Usar correlación manual para replicar exactamente el JS
        lag = 0
        max_corr = -np.inf
        
        for shift in range(-max_lag, max_lag + 1):
            corr = 0
            if shift >= 0:
                if cnt - shift > 0:
                    # self.self.buffer1[i] * self.self.buffer2[i + shift] - Ch2 adelantado respecto Ch1
                    corr = np.sum(buffer1_clean[:cnt - shift] *self.buffer2_clean[shift:cnt])
            else:
                if cnt + shift > 0:
                    # self.self.buffer1[i - shift] * self.self.buffer2[i] - Ch1 adelantado respecto Ch2  
                    corr = np.sum(buffer1_clean[-shift:cnt] *self.buffer2_clean[:cnt + shift])
            
            if corr > max_corr:
                max_corr = corr
                lag = shift
        
        # Assign phase sign - ESTA ES LA PARTE CRÍTICA
        sign = 0
        if lag > 2:
            sign = 1  # Ch2 está adelantado (+)
        elif lag < -2:
            sign = -1  # Ch2 está atrasado (-)
        
        # Cálculo final - CORREGIDO
        # En el JS original: output = sign * phaseDeg - 180
        # Pero esto no tiene sentido físico, vamos a corregirlo
        
        if sign == 0:
            # Sin desfase significativo, determinar por correlación directa
            if sum12 >= 0:
                output = phase_deg  # En fase
            else:
                output = phase_deg - 180  # Fuera de fase
        else:
            # Con desfase significativo
            output = sign * phase_deg
        
        # Normalizar al rango [-180, 180]
        while output > 180:
            output -= 360
        while output < -180:
            output += 360
        
        if _print:
            print(f"fase = {output}")

        self.fase = output

    def _correct_phase(self, Capacitance: float = 2e-9, C1corr = 1, C2corr = 5):
        """
        Corrects fase, must run after calculate fase and power.
        """
        self.fase_corrected = np.rad2deg(np.arctan2(self.V2rms*C2corr * np.sin(np.deg2rad(self.fase)) + (2*np.pi * self.outputFrec * self.V1rms*C1corr * Capacitance),
                                          self.V2rms*C2corr*np.cos(np.deg2rad(self.fase))))
        #print(f"Corr Fase = {self.fase_corrected}")
        #resultado = np.arctan2(Iac * np.sin(phs * np.pi / 180) + (2 * np.pi * (startFrequency + it * freqStep) * Vac * 2.0e-9),Iac * np.cos(phs * np.pi / 180)* 180 / np.pi
        #phs_cor=atan2(Iac*sin(phs*PI/180)+(2*PI*(startFrequency+it*freqStep)*Vac*2.0e-9), Iac*cos(phs*PI/180))*180/PI

    def calculate_Vrms(self):
        """
        Calculates Vrms from chan 1 and 2 buffers.
        """
        # Convertir a arrays si es necesario (una sola vez)
        if not isinstance(self.buffer1, np.ndarray):
            self.buffer1 = np.asarray(self.buffer1, dtype=np.float64)
            self.buffer2 = np.asarray(self.buffer2, dtype=np.float64)
        
        # Stack arrays para procesamiento paralelo
        data_stack = np.stack([self.buffer1, self.buffer2])
                
        # Calcular DC offsets para ambos canales simultáneamente
        dc_offsets = np.mean(data_stack, axis=1)
        
        # Remover DC de ambos canales simultáneamente
        data_clean = data_stack - dc_offsets.reshape(-1, 1)
        
        # Calcular RMS para ambos canales en una sola operación
        rms_values = np.sqrt(np.mean(data_clean * data_clean, axis=1))
        
        # Asignar resultados
        self.V1rms, self.V2rms = rms_values[0], rms_values[1]
        self.dc1, self.dc2 = dc_offsets[0], dc_offsets[1]
        
        return self.V1rms, self.V2rms
    
    def calculate_Power(self, C1corr: int = 1, C2corr: int = 5):
        """
        Calculates Power from the stored V1, V2 (This should be measured and calculated beforehand)
        args:
            - C1corr = correction factor for C1 (no corr by default)
            - C2corr = correction factor for C2 (Itefi corr by default I = Vx5 = V/0.2)
        """
        self.P = (self.V1rms*C1corr * self.V2rms*C2corr)*np.cos(np.deg2rad(self.fase))
    
    def measure_power(self, C1corr = 1, C2corr = 5):
        """
        Default Power measurement for our experiments
        """
        self.measureBuffers()
        self.calculate_phase()
        self.calculate_Vrms()
        self.calculate_Power(C1corr, C2corr)
        #print(f"P = {self.P}")

    def plot_oscilloscope_data(self, title="Oscilloscope Measurement", labels=None, colors=None, dual_scale=True):
        """
        Plotea automáticamente los datos de 1 o 2 canales con escalas independientes.
        
        Parameters:
        - title: título del gráfico
        - labels: lista con etiquetas ['Canal 1', 'Canal 2']
        - colors: lista con colores ['blue', 'red']
        - dual_scale: Si True, usa escalas Y independientes para cada canal
        """
        self.calculate_phase()
        self.calculate_Vrms()
        self.calculate_Power()
        # Verificar datos
        if not hasattr(self, 'buffer1') or self.buffer1 is None:
            print("❌ No hay datos. Ejecuta measureBuffers() primero.")
            return
        
        # Convertir a arrays y verificar tamaños
        buffer1 = np.array(self.buffer1)
        dual_channel = hasattr(self, 'buffer2') and self.buffer2 is not None
        
        if dual_channel:
            buffer2 = np.array(self.buffer2)
            min_size = min(len(buffer1), len(buffer2))
            buffer1, buffer2 = buffer1[:min_size], buffer2[:min_size]
        
        # Configuración por defecto
        labels = labels or (['Canal 1', 'Canal 2'] if dual_channel else ['Canal 1'])
        colors = colors or (['blue', 'red'] if dual_channel else ['blue'])
        
        # Vector de tiempo
        time_total = len(buffer1) / self.sampling_frequency
        if time_total < 1e-3:
            time_scaled = np.linspace(0, time_total * 1e6, len(buffer1))
            time_label = "Tiempo (μs)"
        else:
            time_scaled = np.linspace(0, time_total * 1e3, len(buffer1))
            time_label = "Tiempo (ms)"
        
        # Crear figura
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Canal 1
        ax1.plot(time_scaled, buffer1, color=colors[0], linewidth=1.2, label=labels[0])
        ax1.set_xlabel(time_label, fontsize=12)
        ax1.set_ylabel(f'{labels[0]} (V)', color=colors[0], fontsize=12)
        ax1.tick_params(axis='y', labelcolor=colors[0])
        ax1.grid(True, alpha=0.3)
        
        # Canal 2 con escala independiente si dual_scale=True
        if dual_channel:
            if dual_scale:
                ax2 = ax1.twinx()
                ax2.plot(time_scaled, buffer2, color=colors[1], linewidth=1.2, label=labels[1])
                ax2.set_ylabel(f'{labels[1]} (V)', color=colors[1], fontsize=12)
                ax2.tick_params(axis='y', labelcolor=colors[1])
                
                # Leyenda combinada
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            else:
                # Escala común
                ax1.plot(time_scaled, buffer2, color=colors[1], linewidth=1.2, label=labels[1])
                ax1.set_ylabel('Amplitud (V)', fontsize=12)
                ax1.legend(loc='upper right')
        
        # Título con información técnica
        info = f'Fs: {self.sampling_frequency/1e6:.1f} MHz, Muestras: {len(buffer1)}'
        if hasattr(self, 'V1rms') and hasattr(self, 'V2rms'):
            info += f', RMS: {self.V1rms:.3f}V / {self.V2rms:.3f}V'
        if hasattr(self, 'fase'):
            info += f', Fase: {self.fase:.1f}°'
        
        plt.title(f'{title}\n{info}', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return time_scaled, buffer1, (buffer2 if dual_channel else None)
        #Wavegen control:
    
    def wavegen(self, chan = 1, frec = 35e3, amplitude = 0.25, offset = 0, output_Disable = True):
        # generate a 10KHz sine signal with 2V amplitude on channel 1
        if amplitude > 0.5:
            amplitude = 0.5
        if frec < 1:
            frec = 1
        self.wavegenOn = True
        self.outputFrec = frec
        wavegen.generate(self.device_data, channel=chan, function=wavegen.function.sine, offset=offset, frequency=frec, amplitude=amplitude)
        if output_Disable:
            self.Output_Off(chan)
        else:
            self.Output_On(chan)
        self.pulseFreq = frec
        self.amplitude = amplitude

    def setFrequency(self, freq = 35e3, chan = 1):
        if freq < 1:
            freq = 1
        self.wavegen(frec= freq, amplitude= self.amplitude, output_Disable=False, chan=1)

    def setAmplitude(self, amp = 35e3, chan = 1):
        if amp > 0.5:
            amp = 0.5
        self.wavegen(frec= self.pulseFreq, amplitude= amp, output_Disable=False, chan=1)
     
    def Output_On(self, chan = 1):
        wavegen.enable(self.device_data, channel = chan)

    def Output_Off(self, chan = 1):
        wavegen.disable(self.device_data, channel = chan)

    #run feedback loop:
    @cronometer
    def FeedbackPulse(self, TOn = 1, startfrec = 25e3, startAmplitude = 0.3, c_phase_Obj = 0, P_Obj = 10):
        frec = startfrec
        amp = startAmplitude
        it = 0
        t0 = time.perf_counter()
        self.wavegen(frec=frec, amplitude=amp)
        self.Output_On()
        print("entrando en bucle")
        fase_error = 5
        p_error = 0
        while(time.perf_counter() - t0 < TOn):            
            if not (20e3 <= frec <= 45e3):
                raise ValueError(f"frec fuera de rango: {frec}")
            if not (0.0 <= amp <= 0.5):
                raise ValueError(f"amplitud fuera de rango: {amp}")
            if abs(fase_error) > 3:
                self.setFrequency(frec)
            if abs(p_error) > 0.5 and abs(fase_error) < 10:
                self.setAmplitude(amp) 

            self.measure_power()
            self._correct_phase()    
            
            fase_error = self.fase_corrected - c_phase_Obj
            
            if   fase_error < -50:
                frec += 1250
            elif fase_error < -20:
                frec += 500
            elif fase_error < -10:
                frec += 250
            elif fase_error < -5:
                frec += 100
            elif fase_error < -3:
                pass
            elif fase_error < -1.5:
                pass
            elif fase_error > 50:
                frec -= 1250
            elif fase_error > 20:
                frec -= 500
            elif fase_error > 10:
                frec -= 250
            elif fase_error > 5:
                frec -= 100
            elif fase_error > 3:
                pass
            elif fase_error > 1.5:
                pass

            p_error = self.P - P_Obj
            if abs(fase_error) < 10: #phase amplif only kiks in near resonance 
                if p_error > 0:
                    amp -= 0.01
                else:
                    amp += 0.01

            frec = min(max(frec, 20e3), 45e3)
            amp  = min(max(amp, 0.0), 0.45)

            print(f"frec = {frec}, amp = {amp}, corr fase = {self.fase_corrected}, Vrms = {self.V1rms}, Irms = {self.V2rms*5}, P = {self.P}")
            it = it+1
            time.sleep(0.002)
        self.Output_Off()
        print(f"numero It = {it}")
        print(f"frec = {frec}")
        return float(frec), float(amp)

    def close(self):
        # reset the scope
        self.Output_Off()
        scope.close(self.device_data)

        # reset the wavegen
        wavegen.close(self.device_data)
        device.close(self.device_data)

    def __enter__(self):
        return(self)
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

def Test():
    print("AnalogDiscovery Class Test")
    AD = AnalogDiscovery(chan1Probe=10)
    with AD as AD:
        sF = float(input("InsertStarFrec: \n"))
        print(f"Start Frec = {sF}")
        AD.wavegen(amplitude=0.25, frec=sF)
        AD.Output_On()
        AD.measure_power()
        AD.plot_oscilloscope_data()
        input("Press Enter")
        AD.FeedbackPulse(TOn=10.0, startfrec=sF)
        AD.Output_Off()


if __name__ == "__main__":
    print("AnalogDiscovery Feedback pulse Test:")
    print("45 min test; 540 pulses")
    AD = AnalogDiscovery(chan1Probe=10)
    with AD as AD:
        sF = 30000
        sA = 0.3
        for i in range(540):
            sF, sA = AD.FeedbackPulse(TOn=1, startfrec=sF, startAmplitude=sA)
            AD.Output_Off()
            print(f"LastFrec = {sF}")
            print(f"Pulses Remaining: {540-i}")
            print(f"Time Remaining: {(540-i)*5/60} minutes.")
            time.sleep(4)
    print("Experiment Finished :)")
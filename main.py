import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

# Page configuration
st.set_page_config(
    page_title="Rocket Motor Design Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff4757;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2ed573;
        margin: 1rem 0;
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class Propellant:
    """Enhanced propellant class with additional properties"""
    name: str
    burn_rate_coeff: float  # mm/s
    pressure_exponent: float  # unitless
    density: float  # g/cm³
    characteristic_velocity: float  # m/s
    gamma: float  # ratio of specific heats
    flame_temperature: float  # K
    molecular_weight: float  # g/mol
    color: str  # For visualization

class PropellantDatabase:
    """Enhanced propellant database"""
    def __init__(self):
        self.propellants = {
            'KNDX': Propellant('KNDX', 3.84, 0.688, 1.865, 912, 1.131, 1720, 42.39, '#FF6B6B'),
            'KNSU': Propellant('KNSU', 8.26, 0.319, 1.889, 908, 1.133, 1600, 41.98, '#4ECDC4'),
            'KNER': Propellant('KNER', 5.13, 0.22, 1.820, 895, 1.140, 1650, 43.22, '#45B7D1'),
            'RCandy': Propellant('RCandy', 7.5, 0.12, 1.650, 850, 1.160, 1500, 40.15, '#96CEB4'),
            'APCP': Propellant('APCP', 4.2, 0.35, 1.950, 1580, 1.200, 3200, 30.50, '#FFEAA7'),
            'Black Powder': Propellant('Black Powder', 12.0, 0.8, 1.700, 600, 1.300, 2000, 44.00, '#2D3436')
        }
    
    def get_propellant(self, name: str) -> Propellant:
        return self.propellants.get(name)
    
    def get_propellant_names(self) -> List[str]:
        return list(self.propellants.keys())

class RocketMotorCalculator:
    """Enhanced calculator with advanced features"""
    
    def __init__(self):
        self.prop_db = PropellantDatabase()
        
    def calculate_kn(self, P1: float, prop: Propellant) -> float:
        """Calculate Kn using Formula 1"""
        P1_Pa = P1 * 1e6
        return (P1_Pa**(1 - prop.pressure_exponent) / 
                (prop.burn_rate_coeff * prop.density * prop.characteristic_velocity)) * 1e6
    
    def calculate_cf(self, P1: float, P2: float, k: float) -> float:
        """Calculate thrust coefficient using Formula 2"""
        term1 = (2 * k**2) / (k - 1)
        term2 = (2 / (k + 1))**((k + 1) / (k - 1))
        term3 = 1 - (P2 / P1)**((k - 1) / k)
        return math.sqrt(term1 * term2 * term3)
    
    def calculate_throat_area(self, F: float, CF: float, P1: float) -> float:
        """Calculate throat area using Formula 3"""
        return F / (CF * P1)
    
    def calculate_expansion_ratio(self, P1: float, P2: float, k: float) -> float:
        """Calculate expansion ratio using Formula 5"""
        term_a = ((k + 1) / 2)**(1 / (k - 1))
        term_b = (P2 / P1)**(1 / k)
        term_c = math.sqrt(((k + 1) / (k - 1)) * (1 - (P2 / P1)**((k - 1) / k)))
        inv_epsilon = term_a * term_b * term_c
        return 1 / inv_epsilon
    
    def simulate_thrust_curve(self, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate thrust curve over time"""
        prop = self.prop_db.get_propellant(params['propellant'])
        
        # Time array (0 to burn time)
        burn_time = params['core_length'] / (prop.burn_rate_coeff * 
                   (params['chamber_pressure'] * 1e6)**prop.pressure_exponent / 1000)
        t = np.linspace(0, burn_time, 1000)
        
        # Simplified thrust curve (decreasing with burn area reduction)
        core_radius = params['core_diameter'] / 2
        grain_radius = params['grain_diameter'] / 2
        
        # Calculate burn area over time
        burn_depth = prop.burn_rate_coeff * (params['chamber_pressure'] * 1e6)**prop.pressure_exponent * t / 1000
        current_core_radius = core_radius + burn_depth
        
        # Thrust decreases as core expands
        burn_area = 2 * np.pi * current_core_radius * params['core_length']
        thrust = params['initial_thrust'] * (burn_area / (2 * np.pi * core_radius * params['core_length']))
        
        # Stop when grain is consumed
        mask = current_core_radius < grain_radius
        return t[mask], thrust[mask]

def create_motor_geometry_3d(params: Dict) -> go.Figure:
    """Create 3D visualization of rocket motor geometry"""
    
    # Motor dimensions
    grain_radius = params['grain_diameter'] / 2
    core_radius = params['core_diameter'] / 2
    length = params['core_length']
    
    # Create cylindrical coordinates
    theta = np.linspace(0, 2*np.pi, 50)
    z = np.linspace(0, length, 50)
    THETA, Z = np.meshgrid(theta, z)
    
    # Outer grain surface
    X_outer = grain_radius * np.cos(THETA)
    Y_outer = grain_radius * np.sin(THETA)
    
    # Inner core surface
    X_inner = core_radius * np.cos(THETA)
    Y_inner = core_radius * np.sin(THETA)
    
    fig = go.Figure()
    
    # Add outer surface
    fig.add_trace(go.Surface(
        x=X_outer, y=Y_outer, z=Z,
        colorscale='Reds',
        opacity=0.8,
        name='Propellant Grain',
        showscale=False
    ))
    
    # Add inner surface (core)
    fig.add_trace(go.Surface(
        x=X_inner, y=Y_inner, z=Z,
        colorscale='Blues',
        opacity=0.3,
        name='Core',
        showscale=False
    ))
    
    # Add nozzle throat (simplified)
    throat_radius = math.sqrt(params.get('throat_area', 100) / math.pi)
    nozzle_z = np.array([length, length + 20])
    nozzle_theta = np.linspace(0, 2*np.pi, 20)
    NOZZLE_THETA, NOZZLE_Z = np.meshgrid(nozzle_theta, nozzle_z)
    
    X_nozzle = throat_radius * np.cos(NOZZLE_THETA)
    Y_nozzle = throat_radius * np.sin(NOZZLE_THETA)
    
    fig.add_trace(go.Surface(
        x=X_nozzle, y=Y_nozzle, z=NOZZLE_Z,
        colorscale='Greys',
        opacity=0.9,
        name='Nozzle',
        showscale=False
    ))
    
    fig.update_layout(
        title="3D Rocket Motor Geometry",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600
    )
    
    return fig

def create_parameter_comparison_chart(results: Dict) -> go.Figure:
    """Create radar chart comparing parameters against ideal ranges"""
    
    categories = ['Kn', 'CF', 'Expansion Ratio', 'Chamber Pressure', 'Safety Factor']
    
    # Normalize values to 0-1 scale based on ideal ranges
    values = [
        min(results['kn'] / 235, 1.0),  # Ideal Kn around 235
        min(results['cf'] / 1.5, 1.0),  # Ideal CF around 1.5
        min(results['expansion_ratio'] / 9.5, 1.0),  # Ideal expansion ratio around 9.5
        1.0 - (results['chamber_pressure'] / 10.0),  # Lower pressure is better (safety)
        0.8  # Placeholder safety factor
    ]
    
    ideal_values = [1.0, 1.0, 1.0, 1.0, 1.0]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        name='Current Design',
        line_color='#FF6B6B'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=ideal_values + [ideal_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Ideal Range',
        line_color='#4ECDC4',
        opacity=0.3
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Performance Comparison",
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Rocket Motor Design Studio</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Solid Rocket Motor Analysis & Optimization Platform</p>', unsafe_allow_html=True)
    
    # Initialize calculator
    calc = RocketMotorCalculator()
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown('<h2 class="sub-header">⚙️ Design Parameters</h2>', unsafe_allow_html=True)
        
        # Propellant selection
        propellant_name = st.selectbox(
            "🧪 Propellant Type",
            calc.prop_db.get_propellant_names(),
            index=0
        )
        
        prop = calc.prop_db.get_propellant(propellant_name)
        
        # Display propellant properties
        with st.expander("📊 Propellant Properties"):
            st.metric("Burn Rate Coefficient", f"{prop.burn_rate_coeff} mm/s")
            st.metric("Pressure Exponent", f"{prop.pressure_exponent}")
            st.metric("Density", f"{prop.density} g/cm³")
            st.metric("Characteristic Velocity", f"{prop.characteristic_velocity} m/s")
            st.metric("Gamma (k)", f"{prop.gamma}")
        
        st.markdown("---")
        
        # Main parameters
        st.markdown("### 🎯 Operating Conditions")
        chamber_pressure = st.slider(
            "Chamber Pressure (MPa)",
            min_value=2.5,
            max_value=6.0,
            value=3.5,
            step=0.1,
            help="Initial chamber pressure - affects Kn calculation"
        )
        
        exit_pressure = st.number_input(
            "Exit Pressure (MPa)",
            min_value=0.05,
            max_value=0.2,
            value=0.101325,
            step=0.01,
            help="Atmospheric pressure at nozzle exit"
        )
        
        st.markdown("### 📐 Geometry")
        core_diameter = st.slider(
            "Core Diameter (mm)",
            min_value=5.0,
            max_value=50.0,
            value=20.0,
            step=1.0
        )
        
        core_length = st.slider(
            "Core Length (mm)",
            min_value=50.0,
            max_value=300.0,
            value=150.0,
            step=5.0
        )
        
        grain_diameter = st.slider(
            "Grain Diameter (mm)",
            min_value=core_diameter * 2,
            max_value=100.0,
            value=max(core_diameter * 2.5, 40.0),
            step=1.0,
            help="Must be at least 2x core diameter"
        )
        
        st.markdown("### 🚀 Performance")
        rocket_mass = st.number_input(
            "Rocket Mass (kg)",
            min_value=0.1,
            max_value=100.0,
            value=5.0,
            step=0.1
        )
        
        thrust_to_weight = st.slider(
            "Thrust-to-Weight Ratio",
            min_value=3.0,
            max_value=15.0,
            value=8.0,
            step=0.5
        )
        
        initial_thrust = rocket_mass * 9.81 * thrust_to_weight
        
        # Calculate button
        if st.button("🔥 Calculate Motor Parameters", type="primary"):
            st.session_state.calculate = True
    
    # Main content area
    if 'calculate' in st.session_state and st.session_state.calculate:
        
        # Parameters dictionary
        params = {
            'propellant': propellant_name,
            'chamber_pressure': chamber_pressure,
            'exit_pressure': exit_pressure,
            'core_diameter': core_diameter,
            'core_length': core_length,
            'grain_diameter': grain_diameter,
            'initial_thrust': initial_thrust
        }
        
        # Calculate results
        kn = calc.calculate_kn(chamber_pressure, prop)
        cf = calc.calculate_cf(chamber_pressure, exit_pressure, prop.gamma)
        throat_area = calc.calculate_throat_area(initial_thrust, cf, chamber_pressure)
        throat_diameter = 2 * math.sqrt(throat_area / math.pi)
        expansion_ratio = calc.calculate_expansion_ratio(chamber_pressure, exit_pressure, prop.gamma)
        exit_area = expansion_ratio * throat_area
        exit_diameter = 2 * math.sqrt(exit_area / math.pi)
        
        params['throat_area'] = throat_area
        
        results = {
            'kn': kn,
            'cf': cf,
            'throat_area': throat_area,
            'throat_diameter': throat_diameter,
            'expansion_ratio': expansion_ratio,
            'exit_diameter': exit_diameter,
            'chamber_pressure': chamber_pressure
        }
        
        # Create three columns for layout
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("### 📊 Key Results")
            
            # Check if parameters are in range
            kn_status = "✅" if 200 <= kn <= 270 else "⚠️"
            cf_status = "✅" if 1.0 <= cf <= 2.0 else "⚠️"
            exp_status = "✅" if 7.0 <= expansion_ratio <= 12.0 else "⚠️"
            pressure_status = "✅" if chamber_pressure <= 10.0 else "❌"
            
            st.metric(f"{kn_status} Kn Ratio", f"{kn:.1f}", help="Range: 200-270")
            st.metric(f"{cf_status} Thrust Coefficient", f"{cf:.3f}", help="Range: 1.0-2.0")
            st.metric("🔥 Initial Thrust", f"{initial_thrust:.1f} N")
            st.metric(f"{pressure_status} Chamber Pressure", f"{chamber_pressure:.1f} MPa", help="Max: 10 MPa")
            
        with col2:
            st.markdown("### 🔧 Nozzle Design")
            
            st.metric("Throat Area", f"{throat_area:.2f} mm²")
            st.metric("Throat Diameter", f"{throat_diameter:.2f} mm")
            st.metric(f"{exp_status} Expansion Ratio", f"{expansion_ratio:.1f}", help="Range: 7.0-12.0")
            st.metric("Exit Diameter", f"{exit_diameter:.2f} mm")
            
        with col3:
            st.markdown("### ⚡ Performance Metrics")
            
            # Calculate additional metrics
            burn_area = math.pi * core_diameter * core_length
            mass_flow_rate = (throat_area * chamber_pressure * 1e6) / (prop.characteristic_velocity * 1000)
            specific_impulse = prop.characteristic_velocity * cf / 9.81
            
            st.metric("Burn Area", f"{burn_area:.1f} mm²")
            st.metric("Mass Flow Rate", f"{mass_flow_rate:.3f} kg/s")
            st.metric("Specific Impulse", f"{specific_impulse:.1f} s")
            st.metric("Thrust Duration", f"{(rocket_mass * 0.3) / mass_flow_rate:.1f} s", help="Estimated")
        
        # Safety warnings
        warnings = []
        if not (200 <= kn <= 270):
            warnings.append(f"⚠️ Kn ({kn:.1f}) outside recommended range (200-270)")
        if not (1.0 <= cf <= 2.0):
            warnings.append(f"⚠️ CF ({cf:.3f}) outside recommended range (1.0-2.0)")
        if not (7.0 <= expansion_ratio <= 12.0):
            warnings.append(f"⚠️ Expansion ratio ({expansion_ratio:.1f}) outside recommended range (7.0-12.0)")
        if chamber_pressure > 10.0:
            warnings.append(f"❌ DANGER: Chamber pressure ({chamber_pressure:.1f} MPa) exceeds safety limit!")
        
        if warnings:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("### ⚠️ Design Warnings")
            for warning in warnings:
                st.markdown(f"- {warning}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("### ✅ Design Validation Passed")
            st.markdown("All parameters are within recommended safety ranges!")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualization tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Performance", "🔧 3D Geometry", "📈 Thrust Curve", "📊 Comparison", "📋 Export"])
        
        with tab1:
            # Performance radar chart
            fig_radar = create_parameter_comparison_chart(results)
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Parameter sensitivity analysis
            st.markdown("### 📈 Parameter Sensitivity")
            
            # Create sensitivity analysis
            pressure_range = np.linspace(2.5, 6.0, 20)
            kn_sensitivity = [calc.calculate_kn(p, prop) for p in pressure_range]
            cf_sensitivity = [calc.calculate_cf(p, exit_pressure, prop.gamma) for p in pressure_range]
            
            fig_sensitivity = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Kn vs Chamber Pressure', 'CF vs Chamber Pressure'),
                vertical_spacing=0.1
            )
            
            fig_sensitivity.add_trace(
                go.Scatter(x=pressure_range, y=kn_sensitivity, name='Kn', line=dict(color=prop.color)),
                row=1, col=1
            )
            
            fig_sensitivity.add_trace(
                go.Scatter(x=pressure_range, y=cf_sensitivity, name='CF', line=dict(color='#45B7D1')),
                row=2, col=1
            )
            
            # Add ideal ranges
            fig_sensitivity.add_hline(y=200, line_dash="dash", line_color="green", row=1, col=1)
            fig_sensitivity.add_hline(y=270, line_dash="dash", line_color="green", row=1, col=1)
            fig_sensitivity.add_hline(y=1.0, line_dash="dash", line_color="green", row=2, col=1)
            fig_sensitivity.add_hline(y=2.0, line_dash="dash", line_color="green", row=2, col=1)
            
            fig_sensitivity.update_xaxes(title_text="Chamber Pressure (MPa)")
            fig_sensitivity.update_yaxes(title_text="Kn", row=1, col=1)
            fig_sensitivity.update_yaxes(title_text="CF", row=2, col=1)
            fig_sensitivity.update_layout(height=600, showlegend=False)
            
            st.plotly_chart(fig_sensitivity, use_container_width=True)
        
        with tab2:
            # 3D geometry visualization
            fig_3d = create_motor_geometry_3d(params)
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Cross-section view
            st.markdown("### ✂️ Cross-Section View")
            
            fig_cross = go.Figure()
            
            # Draw motor cross-section
            grain_r = grain_diameter / 2
            core_r = core_diameter / 2
            
            # Outer boundary
            fig_cross.add_shape(
                type="rect",
                x0=-grain_r, y0=0, x1=grain_r, y1=core_length,
                line=dict(color=prop.color, width=3),
                fillcolor=prop.color,
                opacity=0.3
            )
            
            # Core
            fig_cross.add_shape(
                type="rect",
                x0=-core_r, y0=0, x1=core_r, y1=core_length,
                line=dict(color="white", width=2),
                fillcolor="white"
            )
            
            # Nozzle
            throat_r = throat_diameter / 2
            fig_cross.add_shape(
                type="rect",
                x0=-throat_r, y0=core_length, x1=throat_r, y1=core_length + 20,
                line=dict(color="gray", width=2),
                fillcolor="lightgray",
                opacity=0.7
            )
            
            fig_cross.update_layout(
                title="Motor Cross-Section",
                xaxis_title="Radius (mm)",
                yaxis_title="Length (mm)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_cross, use_container_width=True)
        
        with tab3:
            # Thrust curve simulation
            st.markdown("### 🚀 Simulated Thrust Curve")
            
            time_data, thrust_data = calc.simulate_thrust_curve(params)
            
            fig_thrust = go.Figure()
            fig_thrust.add_trace(go.Scatter(
                x=time_data,
                y=thrust_data,
                mode='lines',
                name='Thrust',
                line=dict(color=prop.color, width=3)
            ))
            
            # Add performance metrics
            max_thrust = np.max(thrust_data)
            avg_thrust = np.mean(thrust_data)
            total_impulse = np.trapz(thrust_data, time_data)
            
            fig_thrust.add_hline(y=avg_thrust, line_dash="dash", line_color="orange", 
                               annotation_text=f"Avg: {avg_thrust:.1f} N")
            
            fig_thrust.update_layout(
                title=f"Thrust vs Time (Total Impulse: {total_impulse:.1f} N⋅s)",
                xaxis_title="Time (s)",
                yaxis_title="Thrust (N)",
                height=500
            )
            
            st.plotly_chart(fig_thrust, use_container_width=True)
            
            # Performance summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Max Thrust", f"{max_thrust:.1f} N")
            with col2:
                st.metric("Average Thrust", f"{avg_thrust:.1f} N")
            with col3:
                st.metric("Burn Time", f"{time_data[-1]:.2f} s")
            with col4:
                st.metric("Total Impulse", f"{total_impulse:.1f} N⋅s")
        
        with tab4:
            # Comparison with other propellants
            st.markdown("### 🧪 Propellant Comparison")
            
            comparison_data = []
            for prop_name in calc.prop_db.get_propellant_names():
                test_prop = calc.prop_db.get_propellant(prop_name)
                test_kn = calc.calculate_kn(chamber_pressure, test_prop)
                test_cf = calc.calculate_cf(chamber_pressure, exit_pressure, test_prop.gamma)
                test_isp = test_prop.characteristic_velocity * test_cf / 9.81
                
                comparison_data.append({
                    'Propellant': prop_name,
                    'Kn': test_kn,
                    'CF': test_cf,
                    'Specific Impulse (s)': test_isp,
                    'Density (g/cm³)': test_prop.density,
                    'Flame Temp (K)': test_prop.flame_temperature
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            
            # Highlight current selection
            def highlight_current(s):
                return ['background-color: #FFE5B4' if s.name == propellant_name else '' for _ in s]
            
            st.dataframe(
                df_comparison.style.apply(highlight_current, axis=1),
                use_container_width=True
            )
            
            # Performance comparison chart
            fig_comp = px.scatter(
                df_comparison,
                x='Specific Impulse (s)',
                y='Kn',
                size='Density (g/cm³)',
                color='Propellant',
                title="Propellant Performance Comparison",
                hover_data=['CF', 'Flame Temp (K)']
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
        
        with tab5:
            # Export results
            st.markdown("### 📋 Design Summary & Export")
            
            # Create summary report
            report = {
                "Motor Design Summary": {
                    "Propellant": propellant_name,
                    "Chamber Pressure (MPa)": chamber_pressure,
                    "Core Diameter (mm)": core_diameter,
                    "Core Length (mm)": core_length,
                    "Grain Diameter (mm)": grain_diameter
                },
                "Calculated Results": {
                    "Kn Ratio": round(kn, 2),
                    "Thrust Coefficient": round(cf, 3),
                    "Throat Area (mm²)": round(throat_area, 2),
                    "Throat Diameter (mm)": round(throat_diameter, 2),
                    "Expansion Ratio": round(expansion_ratio, 2),
                    "Exit Diameter (mm)": round(exit_diameter, 2)
                },
                "Performance Metrics": {
                    "Initial Thrust (N)": round(initial_thrust, 1),
                    "Specific Impulse (s)": round(specific_impulse, 1),
                    "Mass Flow Rate (kg/s)": round(mass_flow_rate, 3),
                    "Estimated Burn Time (s)": round((rocket_mass * 0.3) / mass_flow_rate, 2)
                },
                "Safety Assessment": {
                    "Kn in Range": "✅" if 200 <= kn <= 270 else "❌",
                    "CF in Range": "✅" if 1.0 <= cf <= 2.0 else "❌",
                    "Expansion Ratio in Range": "✅" if 7.0 <= expansion_ratio <= 12.0 else "❌",
                    "Pressure Safe": "✅" if chamber_pressure <= 10.0 else "❌"
                }
            }
            
            # Display formatted report
            st.json(report)
            
            # Download buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # JSON download
                json_str = json.dumps(report, indent=2)
                st.download_button(
                    label="📄 Download JSON Report",
                    data=json_str,
                    file_name=f"rocket_motor_design_{propellant_name}_{chamber_pressure}MPa.json",
                    mime="application/json"
                )
            
            with col2:
                # CSV download
                csv_data = []
                for section, data in report.items():
                    for key, value in data.items():
                        csv_data.append({"Category": section, "Parameter": key, "Value": value})
                
                df_export = pd.DataFrame(csv_data)
                csv_str = df_export.to_csv(index=False)
                
                st.download_button(
                    label="📊 Download CSV Data",
                    data=csv_str,
                    file_name=f"rocket_motor_data_{propellant_name}_{chamber_pressure}MPa.csv",
                    mime="text/csv"
                )
            
            with col3:
                # OpenMotor parameters
                openmotor_config = f"""
# OpenMotor Configuration File
# Generated by Rocket Motor Design Studio

[Motor]
Propellant = {propellant_name}
Core Diameter = {core_diameter} mm
Core Length = {core_length} mm
Grain Diameter = {grain_diameter} mm
Throat Diameter = {throat_diameter:.2f} mm
Exit Diameter = {exit_diameter:.2f} mm

[Simulation]
Chamber Pressure = {chamber_pressure} MPa
Expansion Ratio = {expansion_ratio:.2f}
Kn = {kn:.2f}
CF = {cf:.3f}

[Notes]
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Thrust-to-Weight Ratio: {thrust_to_weight}
Rocket Mass: {rocket_mass} kg
"""
                
                st.download_button(
                    label="⚙️ Download OpenMotor Config",
                    data=openmotor_config,
                    file_name=f"openmotor_config_{propellant_name}.txt",
                    mime="text/plain"
                )
            
            # Manufacturing recommendations
            st.markdown("### 🔧 Manufacturing Recommendations")
            
            manufacturing_tips = f"""
            #### Core Casting Recommendations:
            - **Mandrel Diameter**: {core_diameter:.1f} mm
            - **Casting Tube ID**: {grain_diameter:.1f} mm
            - **Recommended Wall Thickness**: {(grain_diameter - core_diameter) / 4:.1f} mm minimum
            
            #### Nozzle Specifications:
            - **Throat Diameter**: {throat_diameter:.2f} mm (±0.1mm tolerance)
            - **Exit Diameter**: {exit_diameter:.2f} mm
            - **Convergent Angle**: 30-45° recommended
            - **Divergent Angle**: 12-18° recommended
            
            #### Safety Considerations:
            - Maximum operating pressure: {chamber_pressure:.1f} MPa
            - Recommended safety factor: 4:1 minimum
            - Case burst pressure requirement: >{chamber_pressure * 4:.1f} MPa
            
            #### Quality Control:
            - Verify throat diameter with pin gauges
            - Check grain density: target {prop.density:.2f} g/cm³
            - Ensure core is concentric and smooth
            """
            
            st.markdown(manufacturing_tips)

    else:
        # Landing page content
        st.markdown("### 🎯 Welcome to the Advanced Rocket Motor Design Studio")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🚀 Features:
            - **Advanced Calculations**: All Team Antariksh formulas implemented
            - **3D Visualization**: Interactive motor geometry
            - **Thrust Curve Simulation**: Predict performance over time
            - **Multi-Propellant Database**: Compare different formulations
            - **Safety Validation**: Automatic range checking
            - **Export Capabilities**: JSON, CSV, OpenMotor configs
            
            #### 📊 Analysis Tools:
            - Parameter sensitivity analysis
            - Performance comparison charts
            - Real-time safety warnings
            - Manufacturing recommendations
            """)
        
        with col2:
            st.markdown("""
            #### 🔬 Supported Propellants:
            - **KNDX** - Potassium Nitrate/Dextrose
            - **KNSU** - Potassium Nitrate/Sucrose  
            - **KNER** - Potassium Nitrate/Erythritol
            - **RCandy** - Rocket Candy
            - **APCP** - Ammonium Perchlorate Composite
            - **Black Powder** - Traditional formulation
            
            #### ⚡ Quick Start:
            1. Select propellant in sidebar
            2. Set operating pressure (2.5-6.0 MPa)
            3. Define motor geometry
            4. Click "Calculate Motor Parameters"
            5. Analyze results in interactive tabs
            """)
        
        # Feature showcase
        st.markdown("### 🎨 Interactive Visualizations")
        
        # Sample visualization
        demo_fig = go.Figure()
        
        # Create sample thrust curve
        t_demo = np.linspace(0, 5, 100)
        thrust_demo = 800 * np.exp(-t_demo/3) * (1 + 0.1 * np.sin(10*t_demo))
        
        demo_fig.add_trace(go.Scatter(
            x=t_demo,
            y=thrust_demo,
            mode='lines',
            name='Sample Thrust Curve',
            line=dict(color='#FF6B6B', width=3)
        ))
        
        demo_fig.update_layout(
            title="Sample Thrust Curve Visualization",
            xaxis_title="Time (s)",
            yaxis_title="Thrust (N)",
            height=400
        )
        
        st.plotly_chart(demo_fig, use_container_width=True)
        
        st.info("👈 **Get Started**: Use the sidebar to input your motor parameters and begin designing!")

# Additional utility functions
def create_deployment_files():
    """Create deployment configuration files"""
    
    # Create requirements.txt content
    requirements_content = """
streamlit>=1.28.0
plotly>=5.15.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
"""
    
    # Create Procfile for Heroku
    procfile_content = "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"
    
    # Create runtime.txt for Heroku
    runtime_content = "python-3.11.6"
    
    # Create streamlit config
    streamlit_config = """
[server]
headless = true
port = $PORT
enableCORS = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
"""
    
    return {
        "requirements.txt": requirements_content,
        "Procfile": procfile_content,
        "runtime.txt": runtime_content,
        ".streamlit/config.toml": streamlit_config
    }

if __name__ == "__main__":
    main()

#ifndef __NATURAL_BRAIN_H_ 
#define __NATURAL_BRAIN_H_ 


#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <thread>
#include <functional>
#include <cmath>
#include <random>
#include <complex>
#include <algorithm>
#include <numeric>


//for matrix: 
#include "../matrix.h"
#include "../utils.h"
#include "../info_helper.h" 
#include "../autoencoder.h"
#include "../rnn.h"
#include "../bit_vector_attribute.h"

#include "../neuralhelper.h" 



//Lobes: Pariental, Temporal, Occipital, Frontal, Limbic ,Occipital, Cerebellum 

//Brain regions: 
//Frontal lobe:
//Prefrontal cortex, Orbitofrontal cortex, Premotor cortex, Primary motor cortex, Broca's area, Frontal eye fields 
//Parietal lobe:
//Primary somatosensory cortex, Somatosensory association cortex, Posterior parietal cortex, Angular gyrus 
//Temporal lobe:
//Primary auditory cortex, Auditory association cortex, Wernicke's area
//Occipital lobe:
//Primary visual cortex, Visual association cortex 
//Limbic lobe:
//Cingulate cortex, Hippocampus, Amygdala, Hypothalamus, Thalamus, Basal ganglia, Nucleus accumbens, Olfactory bulb, Fornix, Mammillary body, Septal nuclei, Pineal gland, Pituitary gland, Parahippocampal gyrus, Dentate gyrus, Subiculum, Paraventricular nucleus, Ventral tegmental area, Nucleus basalis, Locus coeruleus, Raphe nuclei, Reticular formation, Periaqueductal gray, Parabrachial nucleus, Nucleus of the solitary tract, Area postrema, Suprachiasmatic nucleus, Supraoptic nucleus, Arcuate nucleus, Ventromedial nucleus, Dorsomedial nucleus, Lateral hypothalamus, Medial preoptic area, Ventral pallidum, Globus pallidus, Substantia 
//
//Cerebellum:


namespace provallo {

        //basic lobes:
        enum lobes {
            FRONTAL,
            PARIETAL,
            TEMPORAL,
            OCCIPITAL,
            LIMBIC,
            CEREBELLUM
        };  

        class lobe {
            std::string _name;
            std::vector<std::string> _regions;
            std::vector<provallo::matrix<real_t>> _neural_activity; 
            uint64_t lobe_area; //total area fitting the consumption of all the 
            //neural networks in the lobe 
            provallo::neural_net::ptr _neural_network; 
            provallo::auto_encoder<real_t> _auto_encoder; 
            provallo::helmholtz_machine<real_t> _helmholtz_machine; 

            public:
            lobe(const std::string& name) : _name(name), lobe_area(0), _neural_network(nullptr), _auto_encoder(1,1,1), _helmholtz_machine(1, 1,1,1,1,1), _regions() { 
                
            }    
            virtual ~lobe() {
            }
            void add_region(const std::string& region) {
                _regions.push_back(region);
            }
            const std::string& get_name() const {
                return _name;
            }
            const std::vector<std::string>& get_regions() const {
                return _regions;
            }
            void add_neural_activity(const provallo::matrix<real_t>& neural_activity) {
                _neural_activity.push_back(neural_activity);
            } 
            void set_neural_network(provallo::neural_net::ptr nn) {
                _neural_network = nn;
            } 
            void set_auto_encoder(const provallo::auto_encoder<real_t>& ae) {
                _auto_encoder = ae;
            }
            void set_helmholtz_machine(const provallo::helmholtz_machine<real_t>& hm) {
                _helmholtz_machine = hm;
            }
            const provallo::neural_net::ptr get_neural_network() const {
                return _neural_network;
            }
            const provallo::auto_encoder<real_t>& get_auto_encoder() const {
                return _auto_encoder;
            }
            const provallo::helmholtz_machine<real_t>& get_helmholtz_machine() const {
                return _helmholtz_machine;
            }
            const provallo::matrix<real_t>& get_neural_activity(uint64_t i) const {
                return _neural_activity[i];
            }
            uint64_t get_neural_activity_size() const {
                return _neural_activity.size();
            } 
            void set_lobe_area(uint64_t area) {
                lobe_area = area;
            }   
            uint64_t get_lobe_area() const {
                return lobe_area;
            }
            void fit_holzman_machine() {
                _helmholtz_machine.fit(_neural_activity);
            } 
            void fit_auto_encoder(matrix<real_t>& X, matrix<real_t>& Y) {  
                _auto_encoder.fit(X, Y); 
            }
          
        };  

        //reintropic map helper: 
        //gets digital input (images,videos,audio,etc) and maps it to the lobes of the brain 
        //based on the input data [images,videos,audio,etc ]
        class reintropic_map 
        {
            //lobes, can be frontal, parietal, temporal, occipital, limbic, cerebellum[subclasses] 
            //so each lobe type can be mapped to 1..n lobes 

            std::map<lobes,std::vector< std::shared_ptr<lobe> >> _lobes; 
            //spike trains:
            std::vector<std::vector<std::pair<real_t,real_t>>> _spike_trains; 
            //for the brain: 
            std::vector<std::pair<real_t,real_t>> _brain; 
            //for the body: 
            std::vector<std::pair<real_t,real_t>> _body; 
            //for the environment: 
            std::vector<std::pair<real_t,real_t>> _environment; 
            
            std::mutex _lock; 
            real_t lambda_=1.0;
            real_t eta_0=1.0;
            real_t eta=1.0;
            real_t alpha=1.0;
            real_t pi=3.14159265358979323846; 

            //for audio:
            real_t audio_frequency=44100.0; 
            real_t audio_amplitude=1.0; 
            real_t audio_duration=1.0; 
            real_t audio_phase=0.0;
            //for video:
            real_t video_fps=30.0;
            real_t video_duration=1.0;
            real_t video_width=1080.0;
            real_t video_height=720.0;
            //for images:
            real_t image_width=1080.0; 
            real_t image_height=720.0; 
            real_t image_channels=3.0;
            //for text:
            real_t text_length=100.0; 
            real_t text_characters=26.0; 
            //for touch:
            real_t touch_width=1080.0; 
            real_t touch_height=720.0; 
            //for smell: 
            real_t smell_molecules=100.0; 
            //for taste: 
            real_t taste_molecules=100.0; 
            //for proprioception: 
            real_t proprioception_joints=100.0; 
            //for nociception: 
            real_t nociception_pain=100.0; 
            //for thermoception: 
            real_t thermoception_temperature=100.0; 
            //for mechanoception: 
            real_t mechanoception_force=100.0; 
            //for chemoception: 
            real_t chemoception_molecules=100.0; 
            //for thermoception: 
            real_t thermoception_temperature=100.0; 
            //for magnetoreception: 
            real_t magnetoreception_magnetic_field=100.0; 
            //for electroreception: 
            real_t electroreception_electric_field=100.0; 
            //for hygroreception: 
            real_t hygroreception_humidity=100.0; 
            //for baroreception: 
            real_t baroreception_pressure=100.0; 
            //for osmoreception: 
            real_t osmoreception_osmotic_pressure=100.0; 
            //for stretchoreception: 
            real_t stretchoreception_stretch=100.0; 

            //for visual stimuli:
            real_t visual_stimuli_intensity=1.0; 
            real_t visual_stimuli_contrast=1.0; 
            real_t visual_stimuli_color=1.0; 
            real_t visual_stimuli_shape=1.0; 
            real_t visual_stimuli_motion=1.0; 
            real_t visual_stimuli_depth=1.0; 
            real_t visual_stimuli_perspective=1.0; 
            real_t visual_stimuli_light=1.0; 
            real_t visual_stimuli_shadow=1.0; 
            real_t visual_stimuli_reflection=1.0; 
            real_t visual_stimuli_texture=1.0; 
            real_t visual_stimuli_pattern=1.0; 
            real_t visual_stimuli_size=1.0; 
            real_t visual_stimuli_distance=1.0; 
            real_t visual_stimuli_angle=1.0; 
            real_t visual_stimuli_orientation=1.0; 
            real_t visual_stimuli_speed=1.0; 
            real_t visual_stimuli_acceleration=1.0; 
            real_t visual_stimuli_velocity=1.0; 
            real_t visual_stimuli_position=1.0; 
            real_t visual_stimuli_location=1.0; 
            real_t visual_stimuli_center=1.0; 
            real_t visual_stimuli_periphery=1.0; 
            real_t visual_stimuli_focus=1.0; 
            real_t visual_stimuli_blur=1.0; 
            real_t visual_stimuli_sharpness=1.0; 
            real_t visual_stimuli_saturation=1.0; 
            real_t visual_stimuli_brightness=1.0; 
            real_t visual_stimuli_contrast=1.0; 
            real_t visual_stimuli_hue=1.0; 
            
            //for auditory stimuli: 
            real_t auditory_stimuli_frequency=1.0; 
            real_t auditory_stimuli_amplitude=1.0; 
            real_t auditory_stimuli_duration=1.0; 
            real_t auditory_stimuli_phase=1.0; 
            real_t auditory_stimuli_pitch=1.0; 
            real_t auditory_stimuli_timbre=1.0; 
            
            //for olfactory stimuli: 
            real_t olfactory_stimuli_molecules=1.0; 
            real_t olfactory_stimuli_concentration=1.0; 
            real_t olfactory_stimuli_intensity=1.0; 
            real_t olfactory_stimuli_duration=1.0; 
            real_t olfactory_stimuli_phase=1.0; 
            real_t olfactory_stimuli_pitch=1.0; 
            real_t olfactory_stimuli_timbre=1.0; 

            //for gustatory stimuli: 
            real_t gustatory_stimuli_molecules=1.0;  
            real_t gustatory_stimuli_concentration=1.0; 
            real_t gustatory_stimuli_intensity=1.0; 
            real_t gustatory_stimuli_duration=1.0;
            real_t gustatory_stimuli_phase=1.0;
            real_t gustatory_stimuli_pitch=1.0;
            real_t gustatory_stimuli_timbre=1.0;
            
            //for somatosensory stimuli: 
            real_t somatosensory_stimuli_force=1.0; 
            real_t somatosensory_stimuli_pressure=1.0; 
            real_t somatosensory_stimuli_temperature=1.0; 
            real_t somatosensory_stimuli_vibration=1.0; 
            real_t somatosensory_stimuli_itch=1.0; 
            real_t somatosensory_stimuli_pain=1.0; 
            real_t somatosensory_stimuli_touch=1.0; 
            real_t somatosensory_stimuli_tickle=1.0; 
            real_t somatosensory_stimuli_tingle=1.0; 
            real_t somatosensory_stimuli_numb=1.0; 
            real_t somatosensory_stimuli_cold=1.0; 
            real_t somatosensory_stimuli_hot=1.0; 
            real_t somatosensory_stimuli_warm=1.0; 
            real_t somatosensory_stimuli_cool=1.0; 
            real_t somatosensory_stimuli_wet=1.0; 
            real_t somatosensory_stimuli_dry=1.0; 
            real_t somatosensory_stimuli_moist=1.0; 
            real_t somatosensory_stimuli_damp=1.0; 
            real_t somatosensory_stimuli_slippery=1.0; 
            real_t somatosensory_stimuli_sticky=1.0; 
            real_t somatosensory_stimuli_smooth=1.0; 
            real_t somatosensory_stimuli_rough=1.0; 
            real_t somatosensory_stimuli_hard=1.0; 
            real_t somatosensory_stimuli_soft=1.0; 
            real_t somatosensory_stimuli_firm=1.0; 
            real_t somatosensory_stimuli_squishy=1.0; 
            real_t somatosensory_stimuli_spongy=1.0; 
            real_t somatosensory_stimuli_bumpy=1.0; 
            real_t somatosensory_stimuli_lumpy=1.0; 
            //for vestibular stimuli: 
            real_t vestibular_stimuli_acceleration=1.0; 
            real_t vestibular_stimuli_velocity=1.0; 
            real_t vestibular_stimuli_position=1.0; 
            real_t vestibular_stimuli_location=1.0; 
            real_t vestibular_stimuli_center=1.0; 
            real_t vestibular_stimuli_periphery=1.0; 
            real_t vestibular_stimuli_focus=1.0; 
            real_t vestibular_stimuli_blur=1.0; 
            real_t vestibular_stimuli_sharpness=1.0; 
            real_t vestibular_stimuli_saturation=1.0; 
            real_t vestibular_stimuli_brightness=1.0; 
            real_t vestibular_stimuli_contrast=1.0; 
            real_t vestibular_stimuli_hue=1.0; 
            //for proprioceptive stimuli: 
            real_t proprioceptive_stimuli_joints=1.0; 
            real_t proprioceptive_stimuli_muscles=1.0; 
            real_t proprioceptive_stimuli_tendons=1.0; 
            real_t proprioceptive_stimuli_ligaments=1.0; 
            real_t proprioceptive_stimuli_bones=1.0; 
            real_t proprioceptive_stimuli_skin=1.0; 
            real_t proprioceptive_stimuli_hair=1.0; 
            real_t proprioceptive_stimuli_nails=1.0; 
            real_t proprioceptive_stimuli_teeth=1.0; 
            real_t proprioceptive_stimuli_tongue=1.0; 
            real_t proprioceptive_stimuli_lips=1.0; 
            real_t proprioceptive_stimuli_palate=1.0; 
            real_t proprioceptive_stimuli_pharynx=1.0; 
            real_t proprioceptive_stimuli_larynx=1.0; 
            real_t proprioceptive_stimuli_trachea=1.0; 
            real_t proprioceptive_stimuli_bronchi=1.0; 
            real_t proprioceptive_stimuli_lungs=1.0; 
            real_t proprioceptive_stimuli_heart=1.0; 
            real_t proprioceptive_stimuli_stomach=1.0; 
            real_t proprioceptive_stimuli_liver=1.0; 
            real_t proprioceptive_stimuli_gallbladder=1.0; 
            real_t proprioceptive_stimuli_pancreas=1.0; 
            real_t proprioceptive_stimuli_spleen=1.0; 
            real_t proprioceptive_stimuli_kidneys=1.0; 
            real_t proprioceptive_stimuli_adrenal_glands=1.0; 
            real_t proprioceptive_stimuli_thyroid_gland=1.0; 
            real_t proprioceptive_stimuli_parathyroid_glands=1.0; 
            real_t proprioceptive_stimuli_thymus=1.0; 
            real_t proprioceptive_stimuli_pineal_gland=1.0; 
            real_t proprioceptive_stimuli_pituitary_gland=1.0; 
            real_t proprioceptive_stimuli_hypothalamus=1.0; 
            real_t proprioceptive_stimuli_hippocampus=1.0; 
            real_t proprioceptive_stimuli_amygdala=1.0; 
            real_t proprioceptive_stimuli_cingulate_cortex=1.0; 
            real_t proprioceptive_stimuli_hypothalamus=1.0; 
            real_t proprioceptive_stimuli_thalamus=1.0; 
            real_t proprioceptive_stimuli_basal_ganglia=1.0; 
            real_t proprioceptive_stimuli_nucleus_accumbens=1.0; 
            real_t proprioceptive_stimuli_olfactory_bulb=1.0;
            real_t proprioceptive_stimuli_fornix=1.0;
            real_t proprioceptive_stimuli_mammillary_body=1.0;
            real_t proprioceptive_stimuli_septal_nuclei=1.0;
            real_t proprioceptive_stimuli_pineal_gland=1.0; 
            //for nociceptive stimuli: 
            real_t nociceptive_stimuli_pain=1.0; 
            real_t nociceptive_stimuli_pressure=1.0; 
            real_t nociceptive_stimuli_temperature=1.0; 
            real_t nociceptive_stimuli_vibration=1.0; 
            real_t nociceptive_stimuli_itch=1.0; 
            real_t nociceptive_stimuli_touch=1.0; 
            real_t nociceptive_stimuli_tickle=1.0; 
            real_t nociceptive_stimuli_tingle=1.0; 
            std::vector<std::pair<std::complex<real_t>,std::complex<real_t>> > nociceptive_stimuli_location; 
            //for interoceptive stimuli: 
            real_t interoceptive_stimuli_pain=1.0; 
            real_t interoceptive_stimuli_pressure=1.0; 
            real_t interoceptive_stimuli_temperature=1.0; 
            real_t interoceptive_stimuli_vibration=1.0; 
            real_t interoceptive_stimuli_itch=1.0; 
            real_t interoceptive_stimuli_touch=1.0; 
            real_t interoceptive_stimuli_tickle=1.0; 
            real_t interoceptive_stimuli_tingle=1.0; 
            //location of interoceptive stimuli: 
            std::vector<std::pair<real_t,real_t>> interoceptive_stimuli_location; 
            //for exteroceptive stimuli: 
            real_t exteroceptive_stimuli_pain=1.0; 
            real_t exteroceptive_stimuli_pressure=1.0;
            real_t exteroceptive_stimuli_temperature=1.0; 
            real_t exteroceptive_stimuli_vibration=1.0; 
            real_t exteroceptive_stimuli_itch=1.0; 
            real_t exteroceptive_stimuli_touch=1.0; 
            real_t exteroceptive_stimuli_tickle=1.0; 
            real_t exteroceptive_stimuli_tingle=1.0; 
            //location of exteroceptive stimuli:     
            std::vector<std::pair<real_t,real_t>> exteroceptive_stimuli_location; 
            //for enteroceptive stimuli: 
            real_t enteroceptive_stimuli_pain=1.0; 
            real_t enteroceptive_stimuli_pressure=1.0; 
            real_t enteroceptive_stimuli_temperature=1.0; 
            real_t enteroceptive_stimuli_vibration=1.0; 
            real_t enteroceptive_stimuli_itch=1.0; 
            real_t enteroceptive_stimuli_touch=1.0; 
            real_t enteroceptive_stimuli_tickle=1.0; 
            real_t enteroceptive_stimuli_tingle=1.0; 
            //location of enteroceptive stimuli: 
            std::vector<std::pair<std::complex<real_t>,std::complex<real_t>>> enteroceptive_stimuli_location; 
            //for thermoceptive stimuli: 
            real_t thermoceptive_stimuli_pain=1.0; 
            real_t thermoceptive_stimuli_pressure=1.0; 
            real_t thermoceptive_stimuli_temperature=1.0; 
            real_t thermoceptive_stimuli_vibration=1.0; 
            real_t thermoceptive_stimuli_itch=1.0; 
            real_t thermoceptive_stimuli_touch=1.0; 
            real_t thermoceptive_stimuli_tickle=1.0; 
            //location of thermoceptive stimuli: 
            std::vector<std::pair<real_t,real_t>> thermoceptive_stimuli_location; 
            //for mechanoceptive stimuli: 
            real_t mechanoceptive_stimuli_pain=1.0; 
            real_t mechanoceptive_stimuli_pressure=1.0; 
            real_t mechanoceptive_stimuli_temperature=1.0; 
            real_t mechanoceptive_stimuli_vibration=1.0; 
            real_t mechanoceptive_stimuli_itch=1.0; 
            real_t mechanoceptive_stimuli_touch=1.0; 
            real_t mechanoceptive_stimuli_tickle=1.0; 
            //location of mechanoceptive stimuli: 
            std::vector<std::pair<real_t,real_t>> mechanoceptive_stimuli_location; 
            //for chemoceptive stimuli: 
            real_t chemoceptive_stimuli_pain=1.0; 
            real_t chemoceptive_stimuli_pressure=1.0; 
            real_t chemoceptive_stimuli_temperature=1.0; 
            real_t chemoceptive_stimuli_vibration=1.0; 
            real_t chemoceptive_stimuli_itch=1.0; 
            real_t chemoceptive_stimuli_touch=1.0; 
            real_t chemoceptive_stimuli_tickle=1.0; 
            //location of chemoceptive stimuli: 
            std::vector<std::pair<real_t,real_t>> chemoceptive_stimuli_location; 
            //for photoreceptive stimuli: 
            real_t photoreceptive_stimuli_pain=1.0; 
            real_t photoreceptive_stimuli_pressure=1.0; 
            real_t photoreceptive_stimuli_temperature=1.0; 
            real_t photoreceptive_stimuli_vibration=1.0; 
            real_t photoreceptive_stimuli_itch=1.0; 
            real_t photoreceptive_stimuli_touch=1.0; 
            real_t photoreceptive_stimuli_tickle=1.0; 
            //location of photoreceptive stimuli: 
            std::vector<std::pair<real_t,real_t>> photoreceptive_stimuli_location; 
            //for magnetoreceptive stimuli: 
            real_t magnetoreceptive_stimuli_pain=1.0; 
            real_t magnetoreceptive_stimuli_pressure=1.0; 
            real_t magnetoreceptive_stimuli_temperature=1.0; 
            real_t magnetoreceptive_stimuli_vibration=1.0; 
            real_t magnetoreceptive_stimuli_itch=1.0; 
            real_t magnetoreceptive_stimuli_touch=1.0; 
            real_t magnetoreceptive_stimuli_tickle=1.0; 
            //location of magnetoreceptive stimuli:
            std::vector<std::pair<real_t,real_t>> magnetoreceptive_stimuli_location; 
            //for electroreceptive stimuli: 
            real_t electroreceptive_stimuli_pain=1.0; 
            real_t electroreceptive_stimuli_pressure=1.0; 
            real_t electroreceptive_stimuli_temperature=1.0; 
            real_t electroreceptive_stimuli_vibration=1.0; 
            real_t electroreceptive_stimuli_itch=1.0; 
            real_t electroreceptive_stimuli_touch=1.0; 
            real_t electroreceptive_stimuli_tickle=1.0; 
            //location of electroreceptive stimuli: 
            std::vector<std::pair<real_t,real_t>> electroreceptive_stimuli_location; 
            //for hygroreceptive stimuli: 
            real_t hygroreceptive_stimuli_pain=1.0; 
            real_t hygroreceptive_stimuli_pressure=1.0; 
            real_t hygroreceptive_stimuli_temperature=1.0; 
            real_t hygroreceptive_stimuli_vibration=1.0; 
            real_t hygroreceptive_stimuli_itch=1.0; 
            real_t hygroreceptive_stimuli_touch=1.0; 
            real_t hygroreceptive_stimuli_tickle=1.0; 
            //location of hygroreceptive stimuli: 
            std::vector<std::pair<real_t,real_t>> hygroreceptive_stimuli_location; 
            //for baroreceptive stimuli: 
            real_t baroreceptive_stimuli_pain=1.0; 
            real_t baroreceptive_stimuli_pressure=1.0; 
            real_t baroreceptive_stimuli_temperature=1.0; 
            real_t baroreceptive_stimuli_vibration=1.0; 
            real_t baroreceptive_stimuli_itch=1.0; 
            real_t baroreceptive_stimuli_touch=1.0; 
            real_t baroreceptive_stimuli_tickle=1.0; 
            //location of baroreceptive stimuli: 
            std::vector<std::pair<real_t,real_t>> baroreceptive_stimuli_location; 
            //for osmoreceptive stimuli: 
            real_t osmoreceptive_stimuli_pain=1.0; 
            real_t osmoreceptive_stimuli_pressure=1.0; 
            real_t osmoreceptive_stimuli_temperature=1.0; 
            real_t osmoreceptive_stimuli_vibration=1.0; 
            real_t osmoreceptive_stimuli_itch=1.0; 
            real_t osmoreceptive_stimuli_touch=1.0; 
            real_t osmoreceptive_stimuli_tickle=1.0; 
            //location of osmoreceptive stimuli: 
            std::vector<std::pair<real_t,real_t>> osmoreceptive_stimuli_location; 
            //for stretchoreceptive stimuli: 
            real_t stretchoreceptive_stimuli_pain=1.0; 
            real_t stretchoreceptive_stimuli_pressure=1.0; 
            real_t stretchoreceptive_stimuli_temperature=1.0; 
            real_t stretchoreceptive_stimuli_vibration=1.0; 
            real_t stretchoreceptive_stimuli_itch=1.0; 
            real_t stretchoreceptive_stimuli_touch=1.0; 
            real_t stretchoreceptive_stimuli_tickle=1.0;
            //location of stretchoreceptive stimuli: 
            std::vector<std::pair<real_t,real_t>> stretchoreceptive_stimuli_location; 
            //for chemoceptive stimuli: 
            real_t chemoceptive_stimuli_pain=1.0; 
            real_t chemoceptive_stimuli_pressure=1.0; 
            real_t chemoceptive_stimuli_temperature=1.0; 
            real_t chemoceptive_stimuli_vibration=1.0; 
            real_t chemoceptive_stimuli_itch=1.0;
            real_t chemoceptive_stimuli_touch=1.0;
            real_t chemoceptive_stimuli_tickle=1.0;
            //location of chemoceptive stimuli:
            std::vector<std::pair<real_t,real_t>> chemoceptive_stimuli_location; 

            public:
            reintropic_map()
            {

            }   
            virtual ~reintropic_map(){ clear();}


            void add_lobe(lobes lb) {
                std::lock_guard<std::mutex> lock(_lock);
                _lobes[lb] = std::vector<std::shared_ptr<lobe>>(); 
            }
            void remove_lobe(lobes lb) {
                std::lock_guard<std::mutex> lock(_lock);
                _lobes.erase(lb);
            }   
            void add_region(lobes lb, const std::string& region) {
                std::lock_guard<std::mutex> lock(_lock);
                _lobes[lb].push_back(std::make_shared<lobe>(region)); 

            }
            void remove_region(lobes lb, const std::string& region) {
                std::lock_guard<std::mutex> lock(_lock);
                for (auto it = _lobes[lb].begin(); it != _lobes[lb].end(); it++) {
                    if ((*it)->get_name() == region) {
                        
                        _lobes[lb].erase(it);
                        break;
                    }
                }//
            }//   
            void add_neural_activity(lobes lb, const std::string& region, const provallo::matrix<real_t>& neural_activity) {
                std::lock_guard<std::mutex> lock(_lock);
                for (auto it = _lobes[lb].begin(); it != _lobes[lb].end(); it++) {
                    if ((*it)->get_name() == region) {
                        (*it)->add_neural_activity(neural_activity);
                        break;
                    }
                }
            }   
            void set_neural_network(lobes lb, const std::string& region, provallo::neural_net::ptr nn) {
                std::lock_guard<std::mutex> lock(_lock);
                for (auto it = _lobes[lb].begin(); it != _lobes[lb].end(); it++) {
                    if ((*it)->get_name() == region) {
                        (*it)->set_neural_network(nn);
                        break;
                    }
                }
            }   
            void set_auto_encoder(lobes lb, const std::string& region, const provallo::auto_encoder<real_t>& ae) {
                std::lock_guard<std::mutex> lock(_lock);
                for (auto it = _lobes[lb].begin(); it != _lobes[lb].end(); it++) {
                    if ((*it)->get_name() == region) {
                        (*it)->set_auto_encoder(ae);
                        //could have more than one autoencoder per region 

                    }
                }
            }   
            void set_helmholtz_machine(lobes lb, const std::string& region, const provallo::helmholtz_machine<real_t>& hm) {
                std::lock_guard<std::mutex> lock(_lock);
                for (auto it = _lobes[lb].begin(); it != _lobes[lb].end(); it++) {
                    if ((*it)->get_name() == region) {
                        (*it)->set_helmholtz_machine(hm);
                 //       break;
                    }
                }// end for 
            }   //end set_helmholtz_machine 
            const provallo::neural_net::ptr get_neural_network(lobes lb, const std::string& region) const {
                std::lock_guard<std::mutex> lock(_lock);
                for (auto it = this->_lobes[lb].begin(); it != _lobes[lb].end(); it++) {
                    if ((*it)->get_name() == region) {
                        return (*it)->get_neural_network();
                    }   

                }
                return nullptr;
            }   //end get_neural_network 
            const provallo::auto_encoder<real_t>& get_auto_encoder(lobes lb, const std::string& region) const
             {
                std::lock_guard<std::mutex> lock(_lock);
                
                for (auto it = this->_lobes[lb].begin(); it != _lobes[lb].end(); it++) {
                    if ((*it)->get_name() == region) {
                        return (*it)->get_auto_encoder();
                    }   
                }
                std::cerr << "autoencoder not found for region: " << region << std::endl; 
                //return empty autoencoder:
                return auto_encoder<real_t>();                
             }   //end get_auto_encoder

            void clear() {
                std::lock_guard<std::mutex> lock(_lock);
                 for (auto it = _lobes.begin(); it != _lobes.end(); it++) {
                    it->second.clear();  
                  _lobes.clear();

            } //end clear 
            void set_lobe_area(lobes lb, const std::string& region, uint64_t area) {
                std::lock_guard<std::mutex> lock(_lock);
                for (auto it = _lobes[lb].begin(); it != _lobes[lb].end(); it++) {
                    if ((*it)->get_name() == region) {
                        (*it)->set_lobe_area(area);
                        break;
                    }
                }
            } 
            uint64_t get_lobe_area(lobes lb, const std::string& region) const {
                std::lock_guard<std::mutex> lock(_lock);
                uint64_t area = 0ull;
                for (auto it = _lobes[lb].begin(); it != _lobes[lb].end(); it++) {
                    if ((*it)->get_name() == region) {
                        area+= (*it)->get_lobe_area();
                    }
                }
                return area;
            } 
            real_t cortical_magnification_factor(matrix<real_t> stimuli)const
            {
                real_t sum= stimuli.sum(); 
                real_t Y= -lambda_*sum * alpha * pi / (2.0 * eta_0);  
                return Y;
            }
            
            matrix<real_t> auditory_stimuli(matrix<real_t> stimuli) {
                //parse the auditory stimuli sample data:
                //auditory stimuli:
                matrix<real_t> ret = stimuli.transpose(); 
                //process auditory stimuli: 
                //frequency:
                this->audio_frequency = ret.mean();
                this->audio_amplitude = ret.max();
                this->audio_duration += ret.size1();
                this->audio_phase = ret.sum()/ret.size1(); 
                
                //apply sensory gating on the auditory stimuli: 
                
                eta = eta_0 * exp(-lambda_ * ret.sum() * alpha * pi / (2.0 * eta_0)); 

                //apply sensory gating on the auditory stimuli: 
                ret = ret * eta; 
                //amplitude: 
                this->audio_amplitude = ret.max(); 

                //duration:
                this->audio_duration += ret.size1(); 
                //phase:
                this->audio_phase = ret.sum()/ret.size1(); 

                //pitch:
                this->auditory_stimuli_pitch = ret.mean()/this->auditory_stimuli_pitch + 1.0; 

                //timbre:
                this->auditory_stimuli_timbre = ret.mean()/this->auditory_stimuli_timbre + 1.0; 
                
                return ret;
            }
            matrix<real_t> olfactory_stimuli(matrix<real_t> stimuli) {
                //olfactory stimuli: 
                return stimuli;
            } 
            matrix<real_t> gustatory_stimuli(matrix<real_t> stimuli) {
                //gustatory stimuli: 
                return stimuli;
            } 
            matrix<real_t> somatosensory_stimuli(matrix<real_t> stimuli) {
                //somatosensory stimuli: 
                return stimuli;
            }
            matrix<real_t> vestibular_stimuli(matrix<real_t> stimuli) {
                //vestibular stimuli: 
                return stimuli;
            }
            matrix<real_t> proprioceptive_stimuli(matrix<real_t> stimuli) {
                //proprioceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> nociceptive_stimuli(matrix<real_t> stimuli) {
                //nociceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> interoceptive_stimuli(matrix<real_t> stimuli) {
                //interoceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> exteroceptive_stimuli(matrix<real_t> stimuli) {
                //exteroceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> enteroceptive_stimuli(matrix<real_t> stimuli) {
                //enteroceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> thermoceptive_stimuli(matrix<real_t> stimuli) {
                //thermoceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> mechanoceptive_stimuli(matrix<real_t> stimuli) {
                //mechanoceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> chemoreceptive_stimuli(matrix<real_t> stimuli) {
                //chemoreceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> photoreceptive_stimuli(matrix<real_t> stimuli) {
                //photoreceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> magnetoreceptive_stimuli(matrix<real_t> stimuli) {
                //magnetoreceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> electroreceptive_stimuli(matrix<real_t> stimuli) {
                //electroreceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> hygroreceptive_stimuli(matrix<real_t> stimuli) {
                //hygroreceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> baroreceptive_stimuli(matrix<real_t> stimuli) {
                //baroreceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> osmoreceptive_stimuli(matrix<real_t> stimuli) {
                //osmoreceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> stretchoreceptive_stimuli(matrix<real_t> stimuli) {
                //stretchoreceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> chemoceptive_stimuli(matrix<real_t> stimuli) {
                //chemoceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> thermoceptive_stimuli(matrix<real_t> stimuli) {
                //thermoceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> nociceptive_stimuli(matrix<real_t> stimuli) {
                //nociceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> interoceptive_stimuli(matrix<real_t> stimuli) {
                //interoceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> exteroceptive_stimuli(matrix<real_t> stimuli) {
                //exteroceptive stimuli: 
                return stimuli;
            }
            matrix<real_t> enteroceptive_stimuli(matrix<real_t> stimuli) {
                //enteroceptive stimuli: 
                return stimuli;
            }


        }; //end reintropic_map 
        //each lobe is responsible for a set of regions in the brain 
        //
        //frontal lobe is responsible for decision making, planning, and reasoning 
        //behavioral control, and emotions. 
        //It is also responsible for the primary motor cortex, which is involved in the planning and execution of movement. 
        //The frontal lobe is also responsible for the Broca's area, which is involved in speech production. 
        //It holds a generator for the primary motor cortex, Broca's area, and the prefrontal cortex.  
        //The prefrontal cortex is responsible for executive functions, such as decision making, planning, and reasoning. 
        
        
        class frontal_lobe : public lobe {
            
            public:
            frontal_lobe() : lobe("Frontal lobe") {
            }
            virtual ~frontal_lobe() {
            }
            //actions:
            void plan(const matrix<real_t>& data) {
                //create hypothesis based on data: 

            }   
            void reason( std::vector<std::string> reasons) { 
                //reasoning: 

            }   
            void control_behavior() {
                //control behavior: 
            }
            void control_emotions() {
                //control emotions: 
            }
            void execute_movement() {
                //execute movement: 
            }
            void produce_speech() {
                //produce speech: 
            }
            
        };  
        class parietal_lobe : public lobe {
            public:
            parietal_lobe() : lobe("Parietal lobe") {
            }
            virtual ~parietal_lobe() {
            }
        };  
        class temporal_lobe : public lobe {
            public:
            temporal_lobe() : lobe("Temporal lobe") {
            }
            virtual ~temporal_lobe() {
            }
        };  
        //occipital lobe, handles vision, visual processing, and visual memory 
        //primary visual cortex, visual association cortex 
        //visual stimuli are processed in the occipital lobe 

        class bussgang_neural_network : public neural_net {
            public:
            bussgang_neural_network() : neural_net("Bussgang Neural Network") {
            }
            virtual ~bussgang_neural_network() {
            }
            real_t stimulus_correlation(const matrix<real_t>& stimuli) {
                //stimulus correlation: 
                //use bussgang optimal kernel to calculate the correlation between the stimuli 
                real_t correlation = 0.0; 
                return correlation;
            }   

        }; 

        class occipital_lobe : public lobe {
            public:
            occipital_lobe() : lobe("Occipital lobe") {
            }
            virtual ~occipital_lobe() {
            }
        };  
        class limbic_lobe : public lobe {
            public:
            limbic_lobe() : lobe("Limbic lobe") {
            }
            virtual ~limbic_lobe() {
            }
        };  
        class cerebellum : public lobe {
            public:
            cerebellum() : lobe("Cerebellum") {
            }
            virtual ~cerebellum() {
            }
        };  
        //CEREBELLUM is a base class for the cerebellum 
        //balance and coordinations of movements are controlled by the cerebellum 
        //this is parallel to sensory input from the spinal cord 
        //the cerebellum is responsible for the vestibulocerebellum, spinocerebellum, and cerebrocerebellum 
        //each implemented in a separate class 
        class vestibulocerebellum : public cerebellum {
            public:
            vestibulocerebellum() : cerebellum("Vestibulocerebellum") {
            }
            virtual ~vestibulocerebellum() {
            }
        };
        class spinocerebellum : public cerebellum {
            public:
            spinocerebellum() : cerebellum("Spinocerebellum") {
            }
            virtual ~spinocerebellum() {
            }
        };
        class cerebrocerebellum : public cerebellum {
            public:
            cerebrocerebellum() : cerebellum("Cerebrocerebellum") {
            }
            virtual ~cerebrocerebellum() {
            }
        };
        //each lobe is responsible for a set of regions in the brain 

}//end namespace provallo 

//end of file

#endif //DECISION_ENGINE_THEORETICAL_NATURAL_BRAIN_H 
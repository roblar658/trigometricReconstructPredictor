pub mod pipeline;

use std::sync::Arc;

pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    pub async fn init() -> Result<Self, String> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("Kunne ikke hente GPU-adapter: {:?}", e))?;

        let adapter_info = adapter.get_info();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Extrema Classifier Compute Device"),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("Kunne ikke opprette GPU-enhet: {:?}", e))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        })
    }
}

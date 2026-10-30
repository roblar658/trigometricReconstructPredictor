use crate::classification::{ClassificationSummary, ClassifiedExtremum};
use crate::gpu::GpuContext;
use crate::model::{TrigFunction, TrigKind};
use bytemuck::{Pod, Zeroable};
use std::sync::mpsc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct GpuTrigTerm {
    pub freq: f32,
    pub amp: f32,
    pub is_cos: u32, // 0 = sin, 1 = cos
    pub _pad: u32,
}

#[allow(dead_code)]
pub type GpuSineTerm = GpuTrigTerm;

pub const MAX_TERMS: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuParams {
    pub num_terms: u32,
    pub num_samples: u32,
    pub max_extrema: u32,
    pub _pad0: u32,
    pub x_min: f32,
    pub x_max: f32,
    pub _pad1: [f32; 2],
    pub thresholds: [f32; 4],
    pub terms: [GpuTrigTerm; MAX_TERMS],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug)]
pub struct GpuExtremum {
    pub x: f32,
    pub y: f32,
    pub kind: u32,
    pub class_idx: u32,
}

pub struct ExtremaClassifierPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ExtremaClassifierPipeline {
    pub fn new(ctx: &GpuContext) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Extrema Classifier Bind Group Layout"),
                    entries: &[
                        // Binding 0: Uniforms (Params)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Binding 1: Atomic counters (histograms)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Binding 2: Extremum counter
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Binding 3: Extrema output buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Extrema Classifier Pipeline Layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Extrema Classifier Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Kjører GPU-klassifisering for en gitt funksjon over et intervall
    pub fn run(
        &self,
        ctx: &GpuContext,
        function: &TrigFunction,
        x_min: f32,
        x_max: f32,
        num_samples: u32,
        thresholds: [f32; 4],
    ) -> Result<ClassificationSummary, String> {
        let max_extrema = 4096u32;

        let mut gpu_terms = [GpuTrigTerm::default(); MAX_TERMS];
        let num_terms = function.terms.len().min(MAX_TERMS) as u32;
        for (i, t) in function.terms.iter().take(MAX_TERMS).enumerate() {
            gpu_terms[i] = GpuTrigTerm {
                freq: t.freq as f32,
                amp: t.amp as f32,
                is_cos: if t.kind == TrigKind::Cos { 1 } else { 0 },
                _pad: 0,
            };
        }

        let params = GpuParams {
            num_terms,
            num_samples,
            max_extrema,
            _pad0: 0,
            x_min,
            x_max,
            _pad1: [0.0, 0.0],
            thresholds,
            terms: gpu_terms,
        };

        // 1. Alloker GPU-buffere
        let params_buffer =
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Params Uniform Buffer"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let initial_counters = [0u32; 16];
        let counters_buffer =
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Counters Storage Buffer"),
                    contents: bytemuck::cast_slice(&initial_counters),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                });

        let initial_count = [0u32; 1];
        let count_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Extremum Count Storage Buffer"),
                contents: bytemuck::cast_slice(&initial_count),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let extrema_buffer_size =
            (max_extrema as usize * std::mem::size_of::<GpuExtremum>()) as wgpu::BufferAddress;
        let extrema_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Extrema Output Buffer"),
            size: extrema_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 2. Staging-buffere for lesing tilbake til CPU
        let counters_staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Counters Staging Buffer"),
            size: 16 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let count_staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Count Staging Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let extrema_staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Extrema Staging Buffer"),
            size: extrema_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 3. Bind group
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Extrema Classifier Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: counters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: extrema_buffer.as_entire_binding(),
                },
            ],
        });

        // 4. Encode compute pass og kopieringer
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Extrema Classifier Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Extrema Classifier Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (num_samples + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&counters_buffer, 0, &counters_staging, 0, 16 * 4);
        encoder.copy_buffer_to_buffer(&count_buffer, 0, &count_staging, 0, 4);
        encoder.copy_buffer_to_buffer(
            &extrema_buffer,
            0,
            &extrema_staging,
            0,
            extrema_buffer_size,
        );

        ctx.queue.submit(Some(encoder.finish()));

        // 5. Les tilbake fra GPU til CPU
        let (tx, rx) = mpsc::channel();

        let counters_slice = counters_staging.slice(..);
        let tx1 = tx.clone();
        counters_slice.map_async(wgpu::MapMode::Read, move |res| {
            tx1.send((1, res)).unwrap();
        });

        let count_slice = count_staging.slice(..);
        let tx2 = tx.clone();
        count_slice.map_async(wgpu::MapMode::Read, move |res| {
            tx2.send((2, res)).unwrap();
        });

        let extrema_slice = extrema_staging.slice(..);
        let tx3 = tx;
        extrema_slice.map_async(wgpu::MapMode::Read, move |res| {
            tx3.send((3, res)).unwrap();
        });

        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| format!("GPU poll feil: {:?}", e))?;

        for _ in 0..3 {
            let (_, res) = rx.recv().map_err(|e| format!("Mpsc feil: {:?}", e))?;
            res.map_err(|e| format!("Buffer kartleggingsfeil: {:?}", e))?;
        }

        let counters_view = counters_slice
            .get_mapped_range()
            .map_err(|e| format!("Feil ved henting av tellere: {:?}", e))?;
        let raw_counters: &[u32] = bytemuck::cast_slice(&counters_view);
        let mut counters = [0u32; 16];
        counters.copy_from_slice(raw_counters);
        drop(counters_view);
        counters_staging.unmap();

        let count_view = count_slice
            .get_mapped_range()
            .map_err(|e| format!("Feil ved henting av antall: {:?}", e))?;
        let detected_count: u32 = bytemuck::cast_slice(&count_view)[0];
        drop(count_view);
        count_staging.unmap();

        let num_to_read = detected_count.min(max_extrema) as usize;
        let extrema_view = extrema_slice
            .get_mapped_range()
            .map_err(|e| format!("Feil ved henting av ekstrema: {:?}", e))?;
        let raw_extrema: &[GpuExtremum] = bytemuck::cast_slice(&extrema_view);

        let mut extrema = Vec::with_capacity(num_to_read);
        for item in &raw_extrema[..num_to_read] {
            extrema.push(ClassifiedExtremum::new(
                item.x,
                item.y,
                item.kind,
                item.class_idx,
            ));
        }
        drop(extrema_view);
        extrema_staging.unmap();

        // Sorter etter x-koordinat
        extrema.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

        let mut global_max = None;
        let mut global_min = None;

        for ext in &extrema {
            match ext.kind {
                crate::classification::ExtremumKind::Toppunkt => {
                    if let Some((_, max_y)) = global_max {
                        if ext.y > max_y {
                            global_max = Some((ext.x, ext.y));
                        }
                    } else {
                        global_max = Some((ext.x, ext.y));
                    }
                }
                crate::classification::ExtremumKind::Bunnpunkt => {
                    if let Some((_, min_y)) = global_min {
                        if ext.y < min_y {
                            global_min = Some((ext.x, ext.y));
                        }
                    } else {
                        global_min = Some((ext.x, ext.y));
                    }
                }
            }
        }

        let summary = ClassificationSummary {
            total_peaks: counters[10] as usize,
            total_troughs: counters[11] as usize,
            peak_counts: [
                counters[0] as usize,
                counters[1] as usize,
                counters[2] as usize,
                counters[3] as usize,
                counters[4] as usize,
            ],
            trough_counts: [
                counters[5] as usize,
                counters[6] as usize,
                counters[7] as usize,
                counters[8] as usize,
                counters[9] as usize,
            ],
            peak_threshold_exceeded: [
                counters[12] as usize,
                counters[13] as usize,
                counters[14] as usize,
                counters[15] as usize,
            ],
            trough_threshold_exceeded: [0, 0, 0, 0],
            thresholds,
            global_max,
            global_min,
            extrema,
        };

        Ok(summary)
    }
}

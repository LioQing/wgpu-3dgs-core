use std::{io::ErrorKind, num::NonZeroUsize};

use wgpu_3dgs_core::{
    BatchRead, BatchWrite, PlyBatchReader, PlyBatchWriter, PlyGaussianStream, PlyGaussians,
    ProgressiveGaussianRead, ReadIterGaussian, SpzBatchReader, SpzBatchWriter, SpzGaussianShDegree,
    SpzGaussians, SpzGaussiansFromGaussianSliceOptions, SpzGaussiansHeader, SpzGaussiansPositions,
    SpzGaussiansRotations, SpzGaussiansShs, SpzPhase, WriteIterGaussian,
};

use crate::common::given;

fn one() -> NonZeroUsize {
    NonZeroUsize::new(1).unwrap()
}

#[test]
fn ply_stream_delivers_each_gaussian_before_completion() {
    let original = given::ply_gaussians();
    let mut bytes = Vec::new();
    original.write_to(&mut bytes).unwrap();

    let mut stream = PlyGaussianStream::new(bytes.as_slice()).unwrap();
    assert_eq!(stream.total_gaussians(), original.len());

    let mut out = Vec::new();
    assert_eq!(stream.read_gaussians(one(), &mut out).unwrap(), 1);
    assert_eq!(out, original.0[..1]);
    assert!(!ProgressiveGaussianRead::progress(&stream).done);
    assert_eq!(stream.read_gaussians(one(), &mut out).unwrap(), 1);
    assert_eq!(out, original.0);
    assert!(ProgressiveGaussianRead::progress(&stream).done);
    assert_eq!(stream.read_gaussians(one(), &mut out).unwrap(), 0);
}

#[test]
fn ply_batch_round_trip_and_incomplete_finish() {
    let original = given::ply_gaussians();
    let mut bytes = Vec::new();
    let mut writer = PlyBatchWriter::new(&mut bytes, &original).unwrap();
    assert_eq!(writer.progress().total_units, original.len());
    assert_eq!(writer.step(one()).unwrap().completed_units, 1);
    assert_eq!(
        writer.finish().unwrap_err().kind(),
        ErrorKind::UnexpectedEof
    );

    let mut writer = PlyBatchWriter::new(Vec::new(), &original).unwrap();
    while !BatchWrite::progress(&writer).done {
        writer.step(one()).unwrap();
    }
    let bytes = writer.finish().unwrap();
    let loader = PlyBatchReader::new(bytes.as_slice()).unwrap();
    assert_eq!(
        loader.finish().unwrap_err().kind(),
        ErrorKind::UnexpectedEof
    );

    let mut loader = PlyBatchReader::new(bytes.as_slice()).unwrap();
    while !BatchRead::progress(&loader).done {
        loader.step(one()).unwrap();
    }
    assert_eq!(loader.finish().unwrap(), original);
    assert_eq!(
        PlyGaussians::read_from(&mut bytes.as_slice()).unwrap(),
        original
    );
}

#[test]
fn spz_batch_round_trip_all_versions_and_sh_degrees() {
    for version in 1..=3 {
        for degree in 0..=3 {
            let original = SpzGaussians::from_gaussians_with_options(
                given::gaussians(),
                &SpzGaussiansFromGaussianSliceOptions {
                    version,
                    sh_degree: SpzGaussianShDegree::new(degree).unwrap(),
                    ..Default::default()
                },
            )
            .unwrap();
            let writer = SpzBatchWriter::new(Vec::new(), &original).unwrap();
            assert_eq!(
                writer.finish().unwrap_err().kind(),
                ErrorKind::UnexpectedEof
            );
            let mut writer = SpzBatchWriter::new(Vec::new(), &original).unwrap();

            let mut previous = 0;
            while !BatchWrite::progress(&writer).done {
                let progress = writer.step(one()).unwrap();
                assert_eq!(progress.completed_units, previous + 1);
                previous = progress.completed_units;
            }

            assert_eq!(writer.phase(), SpzPhase::Done);
            let bytes = writer.finish().unwrap();

            let reader = SpzBatchReader::new(bytes.as_slice()).unwrap();
            assert_eq!(reader.header(), &original.header);
            assert_eq!(
                reader.finish().unwrap_err().kind(),
                ErrorKind::UnexpectedEof
            );

            let mut reader = SpzBatchReader::new(bytes.as_slice()).unwrap();
            previous = 0;
            while !BatchRead::progress(&reader).done {
                let progress = reader.step(one()).unwrap();
                assert_eq!(progress.completed_units, previous + 1);
                previous = progress.completed_units;
            }
            assert_eq!(reader.finish().unwrap(), original);
        }
    }
}

#[test]
fn spz_batch_finish_rejects_truncated_gzip_trailer() {
    let original = given::spz_gaussians();
    let mut bytes = Vec::new();
    original.write_to(&mut bytes).unwrap();
    bytes.truncate(bytes.len() - 4);

    let mut reader = SpzBatchReader::new(bytes.as_slice()).unwrap();
    while !BatchRead::progress(&reader).done {
        reader.step(one()).unwrap();
    }
    assert!(reader.finish().is_err());
}

#[test]
fn empty_ply_batches_are_already_done() {
    let original = PlyGaussians(Vec::new());
    let bytes = PlyBatchWriter::new(Vec::new(), &original)
        .unwrap()
        .finish()
        .unwrap();
    let reader = PlyBatchReader::new(bytes.as_slice()).unwrap();
    assert!(BatchRead::progress(&reader).done);
    assert_eq!(reader.finish().unwrap(), original);
}

#[test]
fn empty_spz_batches_are_already_done() {
    let original = SpzGaussians {
        header: SpzGaussiansHeader::new(3, 0, SpzGaussianShDegree::new(0).unwrap(), 12, false)
            .unwrap(),
        positions: SpzGaussiansPositions::FixedPoint24(Vec::new()),
        scales: Vec::new(),
        rotations: SpzGaussiansRotations::QuatSmallestThree(Vec::new()),
        alphas: Vec::new(),
        colors: Vec::new(),
        shs: SpzGaussiansShs::Zero,
    };

    let writer = SpzBatchWriter::new(Vec::new(), &original).unwrap();
    assert!(BatchWrite::progress(&writer).done);

    let bytes = writer.finish().unwrap();
    let reader = SpzBatchReader::new(bytes.as_slice()).unwrap();
    assert!(BatchRead::progress(&reader).done);
    assert_eq!(reader.finish().unwrap(), original);
}

#[test]
fn ply_stream_rejects_truncated_record() {
    let original = given::ply_gaussians();
    let mut bytes = Vec::new();
    original.write_to(&mut bytes).unwrap();
    bytes.truncate(bytes.len() - 1);
    let mut stream = PlyGaussianStream::new(bytes.as_slice()).unwrap();

    let mut out = Vec::new();
    assert_eq!(stream.read_gaussians(one(), &mut out).unwrap(), 1);
    assert_eq!(
        stream.read_gaussians(one(), &mut out).unwrap_err().kind(),
        ErrorKind::UnexpectedEof
    );
    assert_eq!(out.len(), 1);
}

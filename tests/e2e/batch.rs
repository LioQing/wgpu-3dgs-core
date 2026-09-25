use std::{io::ErrorKind, num::NonZeroUsize};

use wgpu_3dgs_core::{
    BatchRead, BatchWrite, GaussianStream, Gaussians, GaussiansBatchReader, GaussiansBatchWriter,
    GaussiansSource, GaussiansStream, IterGaussian, PlyBatchReader, PlyBatchWriter,
    PlyGaussianStream, PlyGaussians, ReadIterGaussian, SpzBatchReader, SpzBatchWriter,
    SpzGaussianShDegree, SpzGaussians, SpzGaussiansFromGaussianSliceOptions, SpzGaussiansHeader,
    SpzGaussiansPositions, SpzGaussiansRotations, SpzGaussiansShs, SpzPhase, WriteIterGaussian,
};

use crate::common::given;

fn one() -> NonZeroUsize {
    NonZeroUsize::new(1).unwrap()
}

#[test]
fn test_ply_stream_should_deliver_each_gaussian_before_completion() {
    let original = given::ply_gaussians();
    let mut bytes = Vec::new();
    original.write_to(&mut bytes).unwrap();

    let mut stream = PlyGaussianStream::new(bytes.as_slice()).unwrap();
    assert_eq!(stream.total_gaussians(), original.len());

    let mut out = Vec::new();
    assert_eq!(stream.next().unwrap().unwrap(), original.0[0]);
    assert_eq!(stream.next_batch(one(), &mut out).unwrap(), 1);
    assert_eq!(out, original.0[1..]);
    assert!(GaussianStream::progress(&stream).done);
    assert_eq!(stream.next_batch(one(), &mut out).unwrap(), 0);
    assert!(stream.next().is_none());
}

#[test]
fn test_ply_stream_when_reading_batches_and_single_items_should_share_progress() {
    let original = given::ply_gaussians();
    let mut bytes = Vec::new();
    original.write_to(&mut bytes).unwrap();

    let mut stream = PlyGaussianStream::new(bytes.as_slice()).unwrap();
    let mut out = Vec::new();
    assert_eq!(stream.next_batch(one(), &mut out).unwrap(), 1);
    assert_eq!(out, original.0[..1]);
    assert!(!GaussianStream::progress(&stream).done);
    assert_eq!(stream.next().unwrap().unwrap(), original.0[1]);
    assert_eq!(stream.next_batch(one(), &mut out).unwrap(), 0);
    assert_eq!(out, original.0[..1]);
    assert!(GaussianStream::progress(&stream).done);
    assert!(stream.next().is_none());
}

#[test]
fn test_ply_batch_when_finish_is_incomplete_should_return_error_and_round_trip() {
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
fn test_spz_batch_when_versions_and_sh_degrees_vary_should_round_trip() {
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
fn test_spz_batch_when_gzip_trailer_is_truncated_should_return_error() {
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
fn test_ply_batch_when_empty_should_be_already_done() {
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
fn test_spz_batch_when_empty_should_be_already_done() {
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
fn test_ply_stream_when_record_is_truncated_should_return_error() {
    let original = given::ply_gaussians();
    let mut bytes = Vec::new();
    original.write_to(&mut bytes).unwrap();
    bytes.truncate(bytes.len() - 1);
    let mut stream = PlyGaussianStream::new(bytes.as_slice()).unwrap();

    let mut out = Vec::new();
    assert_eq!(stream.next_batch(one(), &mut out).unwrap(), 1);
    assert_eq!(
        stream.next_batch(one(), &mut out).unwrap_err().kind(),
        ErrorKind::UnexpectedEof
    );
    assert_eq!(out.len(), 1);
    assert!(!stream.progress().done);
    assert!(stream.next().is_none());
}

#[test]
fn test_gaussians_batch_when_source_is_ply_or_spz_should_round_trip() {
    for original in [
        Gaussians::from(given::ply_gaussians()),
        Gaussians::from(given::spz_gaussians()),
    ] {
        let mut writer = GaussiansBatchWriter::new(Vec::new(), &original).unwrap();
        assert!(!writer.progress().done);
        assert_eq!(
            writer.finish().unwrap_err().kind(),
            ErrorKind::UnexpectedEof
        );

        writer = GaussiansBatchWriter::new(Vec::new(), &original).unwrap();
        let total = writer.progress().total_units;
        let mut previous = 0;
        while !writer.progress().done {
            let progress = writer.step(one()).unwrap();
            assert_eq!(progress.completed_units, previous + 1);
            previous = progress.completed_units;
        }
        assert_eq!(previous, total);
        assert!(writer.step(one()).unwrap().done);
        let bytes = writer.finish().unwrap();

        let reader = GaussiansBatchReader::new(bytes.as_slice(), original.source()).unwrap();
        assert_eq!(
            reader.finish().unwrap_err().kind(),
            ErrorKind::UnexpectedEof
        );

        let mut reader = GaussiansBatchReader::new(bytes.as_slice(), original.source()).unwrap();
        previous = 0;
        while !reader.progress().done {
            let progress = reader.step(one()).unwrap();
            assert_eq!(progress.completed_units, previous + 1);
            previous = progress.completed_units;
        }
        assert_eq!(previous, total);
        assert!(reader.step(one()).unwrap().done);
        assert_eq!(reader.finish().unwrap(), original);
    }
}

#[test]
fn test_gaussians_batch_when_source_is_internal_should_return_error() {
    let gaussians = Gaussians::from(given::gaussians());
    assert_eq!(
        GaussiansBatchReader::new(&b""[..], GaussiansSource::Internal)
            .err()
            .unwrap()
            .kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        GaussiansBatchWriter::new(Vec::new(), &gaussians)
            .err()
            .unwrap()
            .kind(),
        ErrorKind::InvalidInput
    );
}

#[test]
fn test_gaussians_stream_when_source_is_ply_should_deliver_unified_gaussians_in_batches() {
    let original = given::ply_gaussians();
    let mut bytes = Vec::new();
    original.write_to(&mut bytes).unwrap();

    let mut stream = GaussiansStream::new(bytes.as_slice(), GaussiansSource::Ply).unwrap();
    assert_eq!(stream.total_gaussians(), original.len());
    let mut out = vec![given::gaussians()[0]];
    for (index, expected) in original.iter_gaussian().enumerate() {
        assert_eq!(stream.next_batch(one(), &mut out).unwrap(), 1);
        assert_eq!(out[index + 1], expected);
        assert_eq!(stream.progress().completed_units, index + 1);
    }
    assert!(stream.progress().done);
    assert_eq!(stream.next_batch(one(), &mut out).unwrap(), 0);
    assert_eq!(out.len(), original.len() + 1);
}

#[test]
fn test_gaussians_stream_when_source_is_spz_or_internal_should_return_error() {
    for source in [GaussiansSource::Spz, GaussiansSource::Internal] {
        let error = GaussiansStream::new(&b""[..], source).err().unwrap();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }
}

#[test]
fn test_gaussians_stream_when_ply_record_is_truncated_should_keep_previous_gaussians() {
    let original = given::ply_gaussians();
    let mut bytes = Vec::new();
    original.write_to(&mut bytes).unwrap();
    bytes.truncate(bytes.len() - 1);
    let mut stream = GaussiansStream::new(bytes.as_slice(), GaussiansSource::Ply).unwrap();
    let mut out = Vec::new();

    assert_eq!(
        stream
            .next_batch(NonZeroUsize::new(original.len()).unwrap(), &mut out)
            .unwrap_err()
            .kind(),
        ErrorKind::UnexpectedEof
    );
    assert_eq!(
        out,
        original
            .iter_gaussian()
            .take(original.len() - 1)
            .collect::<Vec<_>>()
    );
    assert!(!stream.progress().done);
    assert!(stream.next().is_none());
}

#[test]
fn test_gaussians_batch_when_model_is_empty_should_be_already_done() {
    let original = Gaussians::from(PlyGaussians(Vec::new()));
    let writer = GaussiansBatchWriter::new(Vec::new(), &original).unwrap();
    assert!(writer.progress().done);
    let bytes = writer.finish().unwrap();

    let reader = GaussiansBatchReader::new(bytes.as_slice(), GaussiansSource::Ply).unwrap();
    assert!(reader.progress().done);
    assert_eq!(reader.finish().unwrap(), original);
}

#[test]
fn test_gaussians_stream_when_batch_exceeds_remaining_should_append_only_remaining() {
    let original = given::ply_gaussians();
    let mut bytes = Vec::new();
    original.write_to(&mut bytes).unwrap();
    let mut stream = GaussiansStream::new(bytes.as_slice(), GaussiansSource::Ply).unwrap();
    let mut out = Vec::new();

    assert_eq!(
        stream
            .next_batch(NonZeroUsize::new(original.len() + 1).unwrap(), &mut out)
            .unwrap(),
        original.len()
    );
    assert_eq!(out, original.iter_gaussian().collect::<Vec<_>>());
    assert!(stream.progress().done);
}

#[test]
fn test_gaussians_stream_iterator_should_return_unified_gaussians() {
    let original = given::ply_gaussians();
    let mut bytes = Vec::new();
    original.write_to(&mut bytes).unwrap();

    let stream = GaussiansStream::new(bytes.as_slice(), GaussiansSource::Ply).unwrap();
    assert_eq!(
        stream.collect::<std::io::Result<Vec<_>>>().unwrap(),
        original.iter_gaussian().collect::<Vec<_>>()
    );
}

#[test]
fn test_gaussians_stream_iterator_when_record_is_truncated_should_yield_error_once() {
    let original = given::ply_gaussians();
    let mut bytes = Vec::new();
    original.write_to(&mut bytes).unwrap();
    bytes.truncate(bytes.len() - 1);

    let mut stream = GaussiansStream::new(bytes.as_slice(), GaussiansSource::Ply).unwrap();
    assert_eq!(
        stream.next().unwrap().unwrap(),
        original.iter_gaussian().next().unwrap()
    );
    assert_eq!(
        stream.next().unwrap().unwrap_err().kind(),
        ErrorKind::UnexpectedEof
    );
    assert!(stream.next().is_none());
    assert!(!stream.progress().done);
}

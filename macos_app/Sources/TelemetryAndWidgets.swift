import SwiftUI
import AppKit
import AVKit
import UniformTypeIdentifiers
import ObjectiveC.runtime

// MARK: - 实时硬件资源折线图表组件

private func buildSmoothTelemetryPaths(
    samples: [Double?],
    size: CGSize,
    topInset: CGFloat = 5,
    bottomInset: CGFloat = 4
) -> (line: Path, area: Path, points: [CGPoint?]) {
    var line = Path()
    var area = Path()
    guard !samples.isEmpty, size.width > 2, size.height > topInset + bottomInset + 2 else {
        return (line, area, [])
    }

    let usableHeight = size.height - topInset - bottomInset
    let points: [CGPoint?] = samples.enumerated().map { index, sample in
        guard let sample else { return nil }
        let x = samples.count > 1 ? size.width * CGFloat(index) / CGFloat(samples.count - 1) : size.width
        let y = topInset + usableHeight * CGFloat(1 - min(1, max(0, sample)))
        return CGPoint(x: x, y: y)
    }

    // Keep missing intervals open so a driver outage never looks like 0% GPU usage.
    var start = 0
    while start < points.count {
        guard let first = points[start] else { start += 1; continue }
        var segment = Path()
        segment.move(to: first)
        var end = start
        var last = first
        while end + 1 < points.count, let next = points[end + 1] {
            if abs(next.y - last.y) < 0.25 {
                segment.addLine(to: next)
            } else {
                let previous = end > start ? points[end - 1] ?? last : last
                let following = end + 2 < points.count ? points[end + 2] ?? next : next
                let dx = (next.x - last.x) / 3
                let maxY = size.height - bottomInset
                let cp1 = CGPoint(x: last.x + dx, y: min(maxY, max(topInset, last.y + (next.y - previous.y) / 6)))
                let cp2 = CGPoint(x: next.x - dx, y: min(maxY, max(topInset, next.y - (following.y - last.y) / 6)))
                segment.addCurve(to: next, control1: cp1, control2: cp2)
            }
            last = next
            end += 1
        }
        line.addPath(segment)
        if end == start {
            // A single measured point, including the first sample after a gap.
            line.addEllipse(in: CGRect(x: first.x - 1, y: first.y - 1, width: 2, height: 2))
        } else {
            segment.addLine(to: CGPoint(x: last.x, y: size.height))
            segment.addLine(to: CGPoint(x: first.x, y: size.height))
            segment.closeSubpath()
            area.addPath(segment)
        }
        start = end + 1
    }
    return (line, area, points)
}

struct TelemetryGridBackground: View {
    let colorScheme: ColorScheme

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let h = geo.size.height
            let gridColor = colorScheme == .dark
                ? Color.white.opacity(0.075)
                : Color.black.opacity(0.065)

            Path { p in
                for frac in [0.25, 0.50, 0.75] {
                    let y = h * CGFloat(frac)
                    p.move(to: CGPoint(x: 0, y: y))
                    p.addLine(to: CGPoint(x: w, y: y))
                }
                for frac in [0.25, 0.50, 0.75] {
                    let x = w * CGFloat(frac)
                    p.move(to: CGPoint(x: x, y: 0))
                    p.addLine(to: CGPoint(x: x, y: h))
                }
            }
            .stroke(gridColor, style: StrokeStyle(lineWidth: 0.65, dash: [2.5, 2.5]))
        }
    }
}

struct SingleMetricLineChartCard: View {
    let title: String
    var capacity: (used: Double, free: Double, total: Double)? = nil
    let samples: [Double?]
    let accentColor: Color
    let isWarning: Bool
    let isCritical: Bool
    @Environment(\.colorScheme) private var colorScheme
    @State private var hoverLocationX: CGFloat? = nil

    private var effectiveColor: Color {
        if isCritical { return .red }
        if isWarning { return .orange }
        return accentColor
    }

    private func fmt1(_ val: Double) -> String {
        String(format: "%.1f", val)
    }

    var body: some View {
        let palette = StudioPalette(colorScheme: colorScheme)
        let currentPct = capacity.map { min(1, max(0, $0.used / $0.total)) } ?? samples.last ?? nil

        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 5) {
                Circle()
                    .fill(effectiveColor)
                    .frame(width: 6.5, height: 6.5)
                    .shadow(color: effectiveColor.opacity(0.55), radius: 2.5)

                Text(title)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.primary)

                Text(currentPct.map { String(format: "%.0f%%", $0 * 100) } ?? "—")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(effectiveColor)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(
                        Capsule()
                            .fill(effectiveColor.opacity(colorScheme == .dark ? 0.18 : 0.12))
                    )

                Spacer(minLength: 4)

                if let capacity {
                    Text(L(
                        "\(fmt1(capacity.used))/\(fmt1(capacity.total))G · 余\(fmt1(capacity.free))G",
                        "\(fmt1(capacity.used))/\(fmt1(capacity.total))G · \(fmt1(capacity.free))G free"
                    ))
                    .font(.system(size: 10.5, weight: .medium, design: .monospaced))
                    .foregroundStyle(isCritical ? Color.red : (isWarning ? Color.orange : Color.secondary))
                    .lineLimit(1)
                } else {
                    Text(currentPct == nil ? L("暂无读数", "Unavailable") : "0–100%")
                        .font(.system(size: 10.5, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
            }

            GeometryReader { geo in
                let built = buildSmoothTelemetryPaths(samples: samples, size: geo.size)
                let hoverIdx: Int? = {
                    guard let hx = hoverLocationX, geo.size.width > 1, !built.points.isEmpty else { return nil }
                    let ratio = min(1.0, max(0.0, hx / geo.size.width))
                    return min(built.points.count - 1, max(0, Int(round(ratio * CGFloat(built.points.count - 1)))))
                }()

                ZStack(alignment: .topLeading) {
                    RoundedRectangle(cornerRadius: 6.5)
                        .fill(
                            colorScheme == .dark
                                ? Color.black.opacity(0.26)
                                : Color(red: 0.962, green: 0.970, blue: 0.984)
                        )

                    TelemetryGridBackground(colorScheme: colorScheme)
                        .clipShape(RoundedRectangle(cornerRadius: 6.5))

                    built.area
                        .fill(
                            LinearGradient(
                                colors: [
                                    effectiveColor.opacity(colorScheme == .dark ? 0.34 : 0.24),
                                    effectiveColor.opacity(0.02)
                                ],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 6.5))

                    built.line
                        .stroke(
                            effectiveColor,
                            style: StrokeStyle(lineWidth: 1.75, lineCap: .round, lineJoin: .round)
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 6.5))

                    if let hi = hoverIdx, let hp = built.points[hi], let hPct = samples[hi] {
                        let percent = String(format: "%.0f%%", hPct * 100)
                        let reading = capacity.map { "\(fmt1(hPct * $0.total)) GB (\(percent))" } ?? percent

                        Path { p in
                            p.move(to: CGPoint(x: hp.x, y: 0))
                            p.addLine(to: CGPoint(x: hp.x, y: geo.size.height))
                        }
                        .stroke(effectiveColor.opacity(0.55), style: StrokeStyle(lineWidth: 1.0, dash: [2, 2]))

                        Circle()
                            .fill(effectiveColor)
                            .frame(width: 6, height: 6)
                            .overlay(Circle().stroke(Color.white, lineWidth: 1.2))
                            .position(hp)

                        Text(reading)
                            .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.primary)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1.5)
                            .background(
                                RoundedRectangle(cornerRadius: 4)
                                    .fill(palette.workspaceCardFill.opacity(0.94))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 4)
                                            .strokeBorder(effectiveColor.opacity(0.45), lineWidth: 0.7)
                                    )
                            )
                            .position(
                                x: min(max(46, hp.x), geo.size.width - 46),
                                y: 11
                            )
                    } else if let lastPt = built.points.last ?? nil {
                        Circle()
                            .fill(effectiveColor.opacity(0.28))
                            .frame(width: 10, height: 10)
                            .position(lastPt)

                        Circle()
                            .fill(effectiveColor)
                            .frame(width: 4.8, height: 4.8)
                            .position(lastPt)
                    }
                }
                .overlay(
                    RoundedRectangle(cornerRadius: 6.5)
                        .strokeBorder(palette.cardBorder, lineWidth: 0.8)
                )
                .onContinuousHover { phase in
                    switch phase {
                    case .active(let loc):
                        hoverLocationX = loc.x
                    case .ended:
                        hoverLocationX = nil
                    }
                }
            }
            .frame(height: 38)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6.5)
        .background(
            RoundedRectangle(cornerRadius: 9)
                .fill(palette.inputFill)
                .overlay(
                    RoundedRectangle(cornerRadius: 9)
                        .strokeBorder(palette.cardBorder, lineWidth: 0.85)
                )
        )
    }
}

struct CombinedMultiMetricChartCard: View {
    let gpuSamples: [Double?]
    let vramUsed: Double
    let vramTotal: Double
    let vramSamples: [Double?]
    let ramUsed: Double
    let ramTotal: Double
    let ramSamples: [Double?]
    let diskUsed: Double
    let diskTotal: Double
    let diskSamples: [Double?]

    @Environment(\.colorScheme) private var colorScheme
    @State private var hoverX: CGFloat? = nil

    static let gpuColor = Color.purple
    private let vramColor = Color(red: 0.33, green: 0.53, blue: 0.98)
    private let ramColor = Color(red: 0.10, green: 0.76, blue: 0.53)
    private let diskColor = Color(red: 0.05, green: 0.73, blue: 0.88)

    private func fmt1(_ val: Double) -> String {
        String(format: "%.1f", val)
    }

    var body: some View {
        let palette = StudioPalette(colorScheme: colorScheme)
        let vPct = vramTotal > 0 ? min(1.0, max(0.0, vramUsed / vramTotal)) : 0.0
        let rPct = ramTotal > 0 ? min(1.0, max(0.0, ramUsed / ramTotal)) : 0.0
        let dPct = diskTotal > 0 ? min(1.0, max(0.0, diskUsed / diskTotal)) : 0.0


        VStack(alignment: .leading, spacing: 6) {
            GeometryReader { geo in
                let gBuilt = buildSmoothTelemetryPaths(samples: gpuSamples, size: geo.size, topInset: 6, bottomInset: 5)
                let vBuilt = buildSmoothTelemetryPaths(samples: vramSamples, size: geo.size, topInset: 6, bottomInset: 5)
                let rBuilt = buildSmoothTelemetryPaths(samples: ramSamples, size: geo.size, topInset: 6, bottomInset: 5)
                let dBuilt = buildSmoothTelemetryPaths(samples: diskSamples, size: geo.size, topInset: 6, bottomInset: 5)

                ZStack(alignment: .topLeading) {
                    RoundedRectangle(cornerRadius: 7)
                        .fill(
                            colorScheme == .dark
                                ? Color.black.opacity(0.28)
                                : Color(red: 0.962, green: 0.970, blue: 0.984)
                        )

                    TelemetryGridBackground(colorScheme: colorScheme)
                        .clipShape(RoundedRectangle(cornerRadius: 7))

                    VStack {
                        Text("100%")
                        Spacer()
                        Text("50%")
                        Spacer()
                        Text("0%")
                    }
                    .font(.system(size: 8.5, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary.opacity(0.65))
                    .padding(.leading, 4)
                    .padding(.vertical, 3)

                    vBuilt.area
                        .fill(
                            LinearGradient(
                                colors: [vramColor.opacity(0.18), vramColor.opacity(0.01)],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 7))

                    if diskTotal > 0 {
                        dBuilt.line
                            .stroke(diskColor, style: StrokeStyle(lineWidth: 1.55, lineCap: .round, lineJoin: .round))
                            .clipShape(RoundedRectangle(cornerRadius: 7))
                    }
                    if ramTotal > 0 {
                        rBuilt.line
                            .stroke(ramColor, style: StrokeStyle(lineWidth: 1.65, lineCap: .round, lineJoin: .round))
                            .clipShape(RoundedRectangle(cornerRadius: 7))
                    }
                    if vramTotal > 0 {
                        vBuilt.line
                            .stroke(vramColor, style: StrokeStyle(lineWidth: 1.85, lineCap: .round, lineJoin: .round))
                            .clipShape(RoundedRectangle(cornerRadius: 7))
                    }

                    gBuilt.line
                        .stroke(Self.gpuColor, style: StrokeStyle(lineWidth: 1.85, lineCap: .round, lineJoin: .round))
                        .clipShape(RoundedRectangle(cornerRadius: 7))

                    if let hx = hoverX, geo.size.width > 1 {
                        let clampedX = min(geo.size.width, max(0, hx))
                        Path { p in
                            p.move(to: CGPoint(x: clampedX, y: 0))
                            p.addLine(to: CGPoint(x: clampedX, y: geo.size.height))
                        }
                        .stroke(Color.primary.opacity(0.32), style: StrokeStyle(lineWidth: 1.0, dash: [2.5, 2.5]))
                    } else {
                        if let lp = gBuilt.points.last ?? nil {
                            Circle().fill(Self.gpuColor).frame(width: 5, height: 5).position(lp)
                        }
                        if vramTotal > 0, let lp = vBuilt.points.last ?? nil {
                            Circle().fill(vramColor).frame(width: 5, height: 5).position(lp)
                        }
                        if ramTotal > 0, let lp = rBuilt.points.last ?? nil {
                            Circle().fill(ramColor).frame(width: 5, height: 5).position(lp)
                        }
                        if diskTotal > 0, let lp = dBuilt.points.last ?? nil {
                            Circle().fill(diskColor).frame(width: 5, height: 5).position(lp)
                        }
                    }
                }
                .overlay(
                    RoundedRectangle(cornerRadius: 7)
                        .strokeBorder(palette.cardBorder, lineWidth: 0.8)
                )
                .onContinuousHover { phase in
                    switch phase {
                    case .active(let loc):
                        hoverX = loc.x
                    case .ended:
                        hoverX = nil
                    }
                }
            }
            .frame(height: 88)

            VStack(spacing: 3.5) {
                legendRow(color: Self.gpuColor, label: L("GPU 利用率", "GPU utilization"), pct: gpuSamples.last ?? nil)
                if vramTotal > 0 {
                    legendRow(color: vramColor, label: L("显存 VRAM", "VRAM"), pct: vPct, capacity: (vramUsed, vramTotal))
                }
                if ramTotal > 0 {
                    legendRow(color: ramColor, label: L("内存 RAM", "System RAM"), pct: rPct, capacity: (ramUsed, ramTotal))
                }
                if diskTotal > 0 {
                    legendRow(color: diskColor, label: L("硬盘 Disk", "Disk"), pct: dPct, capacity: (diskUsed, diskTotal))
                }
            }
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 9)
                .fill(palette.inputFill)
                .overlay(
                    RoundedRectangle(cornerRadius: 9)
                        .strokeBorder(palette.cardBorder, lineWidth: 0.85)
                )
        )
    }

    @ViewBuilder
    private func legendRow(color: Color, label: String, pct: Double?, capacity: (used: Double, total: Double)? = nil) -> some View {
        HStack(spacing: 5) {
            RoundedRectangle(cornerRadius: 2)
                .fill(color)
                .frame(width: 8, height: 3.5)
            Text(label)
                .font(.system(size: 10.5, weight: .medium))
                .foregroundStyle(.secondary)
            Spacer()
            if let capacity {
                Text("\(fmt1(capacity.used)) / \(fmt1(capacity.total)) GB")
                    .font(.system(size: 10.5, weight: .medium, design: .monospaced))
                    .foregroundStyle(.primary)
            }
            Text(pct.map { String(format: "%3.0f%%", $0 * 100) } ?? "—")
                .font(.system(size: 10.5, weight: .bold, design: .monospaced))
                .foregroundStyle(color)
                .frame(width: 34, alignment: .trailing)
        }
    }
}

// MARK: - 图片拖拽与选择组件

struct FrameDropWell: View {
    let title: String
    let subtitle: String
    @Binding var image: NSImage?
    @Environment(\.colorScheme) private var colorScheme
    @State private var isTargeted: Bool = false

    var body: some View {
        let palette = StudioPalette(colorScheme: colorScheme)
        ZStack {
            RoundedRectangle(cornerRadius: 10)
                .fill(palette.inputFill)
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(
                            isTargeted ? Color.accentColor : palette.cardBorder,
                            style: StrokeStyle(lineWidth: isTargeted ? 2.0 : 1.2, dash: [6])
                        )
                )

            if let img = image {
                Image(nsImage: img)
                    .resizable()
                    .scaledToFit()
                    .cornerRadius(8)
                    .padding(6)
                    .overlay(alignment: .topTrailing) {
                        Button {
                            image = nil
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.title3)
                                .foregroundStyle(.white, .black.opacity(0.65))
                        }
                        .buttonStyle(.plain)
                        .padding(10)
                    }
            } else {
                VStack(spacing: 6) {
                    Image(systemName: "photo.badge.plus")
                        .font(.system(size: 24, weight: .medium))
                        .foregroundStyle(Color.accentColor)
                    Text(title)
                        .font(.subheadline.weight(.semibold))
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Button(L("选择本地图片", "Choose Image")) {
                        pickImage()
                    }
                    .controlSize(.small)
                    .padding(.top, 2)
                }
                .padding(12)
            }
        }
        .frame(height: 155)
        .onDrop(of: [.fileURL, .image], isTargeted: $isTargeted) { providers in
            handleDrop(providers: providers)
        }
    }

    private func pickImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url, let loaded = NSImage(contentsOf: url) {
            image = loaded
        }
    }

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        if provider.canLoadObject(ofClass: URL.self) {
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                if let url = url, let loaded = NSImage(contentsOf: url) {
                    DispatchQueue.main.async { self.image = loaded }
                }
            }
            return true
        }
        return false
    }
}


struct EditReferencePicker: View {
    @Binding var images: [NSImage]
    @Binding var importing: Bool
    let limit: Int
    @State private var targeted = false
    @State private var importError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !images.isEmpty {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 100, maximum: 150))], alignment: .leading) {
                    ForEach(Array(images.enumerated()), id: \.offset) { index, image in
                        VStack(spacing: 4) {
                            Image(nsImage: image)
                                .resizable().scaledToFit().frame(height: 68)
                                .accessibilityLabel(L("参考图 \(index + 1)", "Reference \(index + 1)"))
                            HStack(spacing: 5) {
                                Text("\(index + 1)").monospacedDigit()
                                Spacer()
                                Button { images.swapAt(index, index - 1) } label: { Image(systemName: "chevron.left") }
                                    .disabled(index == 0)
                                    .help(L("前移", "Move earlier"))
                                Button { images.swapAt(index, index + 1) } label: { Image(systemName: "chevron.right") }
                                    .disabled(index == images.count - 1)
                                    .help(L("后移", "Move later"))
                                Button { images.remove(at: index); importError = nil } label: { Image(systemName: "xmark.circle") }
                                    .help(L("移除", "Remove"))
                            }
                            .buttonStyle(.borderless)
                            .font(.caption)
                        }
                        .padding(6)
                        .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                    }
                }
            }
            HStack(spacing: 10) {
                Button(L("添加参考图…", "Add References…")) {
                    let panel = NSOpenPanel()
                    panel.allowedContentTypes = [.image]
                    panel.allowsMultipleSelection = limit > 1
                    if panel.runModal() == .OK, accepts(panel.urls.count) {
                        append(panel.urls.map { NSImage(contentsOf: $0) })
                    }
                }
                PasteButton(supportedContentTypes: [.fileURL, .image]) { providers in
                    _ = importProviders(providers)
                }
                if importing {
                    ProgressView().controlSize(.small)
                } else {
                    Text(L("或拖入图片", "or drop images here")).foregroundStyle(.secondary)
                }
            }
            .controlSize(.small)
            .disabled(importing || images.count >= limit)
            if images.count > limit {
                Text(L("请移除 \(images.count - limit) 张图片后生成。", "Remove \(images.count - limit) images before generating."))
                    .font(.caption).foregroundStyle(.orange)
            } else if let importError {
                Text(importError).font(.caption).foregroundStyle(.orange)
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(targeted ? Color.accentColor : .clear))
        .onDrop(of: [.fileURL, .image], isTargeted: $targeted, perform: importProviders)
    }

    private func accepts(_ count: Int) -> Bool {
        guard images.count + count <= limit else {
            importError = L("本次最多还可添加 \(max(0, limit - images.count)) 张参考图。",
                            "You can add up to \(max(0, limit - images.count)) more references.")
            return false
        }
        return true
    }

    private func append(_ batch: [NSImage?]) {
        let valid = batch.compactMap { $0 }
        guard valid.count == batch.count else {
            importError = L("部分图片无法读取，请重新选择。", "Some images could not be read. Please select them again.")
            return
        }
        images.append(contentsOf: valid)
        importError = nil
    }

    private func importProviders(_ providers: [NSItemProvider]) -> Bool {
        guard !providers.isEmpty, !importing, accepts(providers.count) else { return false }
        importing = true
        Task { @MainActor in
            defer { importing = false }
            var batch: [NSImage?] = []
            for provider in providers {
                if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
                    let url: URL? = await withCheckedContinuation { continuation in
                        _ = provider.loadObject(ofClass: URL.self) { url, _ in continuation.resume(returning: url) }
                    }
                    batch.append(url.flatMap { NSImage(contentsOf: $0) })
                } else if let imageType = provider.registeredTypeIdentifiers.first(where: {
                    UTType($0)?.conforms(to: .image) == true
                }) {
                    let data: Data? = await withCheckedContinuation { continuation in
                        provider.loadDataRepresentation(forTypeIdentifier: imageType) { data, _ in
                            continuation.resume(returning: data)
                        }
                    }
                    batch.append(data.flatMap { NSImage(data: $0) })
                } else {
                    batch.append(nil)
                }
            }
            append(batch)
        }
        return true
    }
}


// MARK: - 通用 UI 容器组件

struct SidebarCard<Content: View>: View {
    let title: String
    let icon: String
    var badge: String? = nil
    @Environment(\.colorScheme) private var colorScheme
    @ViewBuilder var content: () -> Content

    var body: some View {
        let palette = StudioPalette(colorScheme: colorScheme)
        VStack(alignment: .leading, spacing: 11) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Color.accentColor)
                Text(title)
                    .font(.system(size: 12.5, weight: .bold))
                    .foregroundStyle(.primary)
                Spacer()
                if let badge {
                    Text(badge)
                        .font(.system(size: 10.5, weight: .medium))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.primary.opacity(0.06)))
                }
            }

            content()
        }
        .padding(13)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(palette.sidebarCardFill)
                .shadow(color: .black.opacity(colorScheme == .dark ? 0.22 : 0.04), radius: 3, x: 0, y: 1)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(palette.cardBorder, lineWidth: 0.8)
        )
    }
}

struct WorkspaceCard<Content: View>: View {
    let title: String
    var subtitle: String? = nil
    var icon: String? = nil
    var trailing: AnyView? = nil
    @Environment(\.colorScheme) private var colorScheme
    @ViewBuilder var content: () -> Content

    var body: some View {
        let palette = StudioPalette(colorScheme: colorScheme)
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .center, spacing: 8) {
                if let icon {
                    Image(systemName: icon)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(Color.accentColor)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.headline)
                    if let subtitle {
                        Text(subtitle).font(.caption).foregroundStyle(.secondary)
                    }
                }
                Spacer()
                if let trailing { trailing }
            }
            content()
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 11)
                .fill(palette.workspaceCardFill)
                .shadow(color: .black.opacity(colorScheme == .dark ? 0.25 : 0.035), radius: 4, x: 0, y: 1.5)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 11)
                .strokeBorder(palette.cardBorder, lineWidth: 0.8)
        )
    }
}

struct StatusDot: View {
    let color: Color
    var body: some View {
        Circle().fill(color).frame(width: 8, height: 8)
    }
}

func fmt(_ v: Double, _ digits: Int = 2) -> String {
    String(format: "%.\(digits)f", v)
}


extension View {
    @ViewBuilder
    func compatScrollIndicatorMargins(_ edges: Edge.Set = .all, _ length: CGFloat) -> some View {
        if #available(macOS 14.0, *) {
            self.contentMargins(edges, length, for: .scrollIndicators)
        } else {
            self
        }
    }

    @ViewBuilder
    func compatOnChange<V: Equatable>(of value: V, perform action: @escaping (V) -> Void) -> some View {
        if #available(macOS 14.0, *) {
            self.onChange(of: value) { _, newValue in
                action(newValue)
            }
        } else {
            self.onChange(of: value, perform: action)
        }
    }
}

// MARK: - 极简悬浮滚动条引擎（消除外接鼠标下的传统宽灰槽滚动条）

extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}

@MainActor
enum SleekScrollBarInstaller {
    private static var installed = false

    static func install() {
        guard !installed else { return }
        installed = true

        if let orig = class_getClassMethod(NSScroller.self, #selector(getter: NSScroller.preferredScrollerStyle)),
           let repl = class_getClassMethod(NSScroller.self, #selector(NSScroller.sleek_preferredScrollerStyle)) {
            method_exchangeImplementations(orig, repl)
        }

        if let origSet = class_getInstanceMethod(NSScrollView.self, #selector(setter: NSScrollView.scrollerStyle)),
           let replSet = class_getInstanceMethod(NSScrollView.self, #selector(NSScrollView.sleek_setScrollerStyle(_:))) {
            method_exchangeImplementations(origSet, replSet)
        }

        if let origTile = class_getInstanceMethod(NSScrollView.self, #selector(NSScrollView.tile)),
           let replTile = class_getInstanceMethod(NSScrollView.self, #selector(NSScrollView.sleek_tile)) {
            method_exchangeImplementations(origTile, replTile)
        }
    }
}

extension NSScroller {
    @objc fileprivate static func sleek_preferredScrollerStyle() -> NSScroller.Style {
        return .overlay
    }
}

extension NSScrollView {
    @objc fileprivate func sleek_setScrollerStyle(_ newStyle: NSScroller.Style) {
        self.sleek_setScrollerStyle(.overlay)
    }

    @objc fileprivate func sleek_tile() {
        if scrollerStyle != .overlay {
            self.sleek_setScrollerStyle(.overlay)
        }
        if !autohidesScrollers {
            autohidesScrollers = true
        }
        if let v = verticalScroller, v.controlSize != .small {
            v.controlSize = .small
        }
        if let h = horizontalScroller, h.controlSize != .small {
            h.controlSize = .small
        }

        let isDark = effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
        let desiredKnob: NSScroller.KnobStyle = isDark ? .light : .default
        if scrollerKnobStyle != desiredKnob {
            scrollerKnobStyle = desiredKnob
        }

        if let textView = documentView as? NSTextView {
            scrollerInsets = NSEdgeInsets(top: 3, left: 0, bottom: 3, right: 1)
            if textView.textContainerInset.height < 3 {
                textView.textContainerInset = NSSize(width: 2, height: 4)
            }
        }

        self.sleek_tile()
    }
}

// MARK: - 窗口顶栏拖拽与红绿灯对齐组件

struct WindowDragHandle: NSViewRepresentable {
    final class DragView: NSView {
        override var mouseDownCanMoveWindow: Bool { true }
    }

    func makeNSView(context: Context) -> NSView {
        DragView()
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}

struct TitlebarConfigurator: NSViewRepresentable {
    let headerHeight: CGFloat

    final class ConfigView: NSView {
        var headerHeight: CGFloat = 40
        private var resizeObserver: NSObjectProtocol?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            configureWindow()
            if let resizeObserver {
                NotificationCenter.default.removeObserver(resizeObserver)
            }
            if let win = window {
                resizeObserver = NotificationCenter.default.addObserver(
                    forName: NSWindow.didResizeNotification,
                    object: win,
                    queue: .main
                ) { [weak self] _ in
                    self?.alignTrafficLights()
                }
            }
        }

        override func layout() {
            super.layout()
            alignTrafficLights()
        }

        private func configureWindow() {
            guard let win = window else { return }
            win.titleVisibility = .hidden
            win.titlebarAppearsTransparent = true
            win.styleMask.insert(.fullSizeContentView)
            win.isMovableByWindowBackground = false
            alignTrafficLights()
        }

        func alignTrafficLights() {
            guard let win = window else { return }
            let types: [NSWindow.ButtonType] = [.closeButton, .miniaturizeButton, .zoomButton]
            for (idx, bType) in types.enumerated() {
                guard let btn = win.standardWindowButton(bType),
                      let superview = btn.superview else { continue }
                var frame = btn.frame
                frame.origin.x = 15.0 + CGFloat(idx) * 20.0
                if superview.isFlipped {
                    frame.origin.y = (headerHeight - frame.height) * 0.5
                } else {
                    frame.origin.y = superview.bounds.height - (headerHeight + frame.height) * 0.5
                }
                btn.setFrameOrigin(frame.origin)
            }
        }

        deinit {
            if let resizeObserver {
                NotificationCenter.default.removeObserver(resizeObserver)
            }
        }
    }

    func makeNSView(context: Context) -> ConfigView {
        let v = ConfigView()
        v.headerHeight = headerHeight
        return v
    }

    func updateNSView(_ nsView: ConfigView, context: Context) {
        nsView.headerHeight = headerHeight
        nsView.alignTrafficLights()
    }
}


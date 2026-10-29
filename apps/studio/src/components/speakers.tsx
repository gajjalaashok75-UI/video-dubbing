/* ===========================================================================
   The detected speakers, and the voice each one was cloned from.

   Built from the same console primitives as the rest of the inspector, so it
   reads as part of the studio rather than a panel bolted on.

   The design problem here is that a bad clone is invisible. A dub where one
   actor came out in a generic voice sounds like a mediocre dub, not like a
   bug — so the state and the warning are the most prominent things on the
   row, and a speaker that needs attention is impossible to miss.

   Two rules the component holds:

   * the internal label is always shown, even when an alias is set, because
     that is the id every other part of the system uses;
   * changing one speaker's reference says what it costs, and it costs only
     that speaker's phrases.
   =========================================================================== */

import { useState } from "react";
import {
  User, Warning, CheckCircle, Question, Waveform,
} from "@phosphor-icons/react";
import { Sub, Tool, Row, Lamp } from "./console";
import { cx } from "@/lib/cx";
import {
  chooseReference, getReferenceWindows, renameSpeaker, speakerStatusLabel,
  artifactUrl,
  type ReferenceWindow, type SpeakerProfile,
} from "@/lib/api";

/**
 * A speaker's status, in the studio's own lamp vocabulary.

 * `Lamp` knows idle, run, good and bad. A short reference is neither a failure
 * nor a success, and `run` is the accent this studio already uses for
 * "attention, but working" — so it goes there rather than inventing a state.
 */
function tone(status: SpeakerProfile["status"]) {
  if (status === "cloned") return "good" as const;
  if (status === "short_reference") return "run" as const;
  return "bad" as const;
}

function StatusIcon({ status }: { status: SpeakerProfile["status"] }) {
  if (status === "cloned") return <CheckCircle size={11} weight="fill" />;
  if (status === "short_reference") return <Warning size={11} weight="fill" />;
  if (status === "generic_fallback") return <Question size={11} weight="fill" />;
  return <Warning size={11} weight="fill" />;
}

function seconds(n?: number | null) {
  return n == null ? "—" : `${n.toFixed(1)}s`;
}

export function Speakers({
  jobId,
  speakers,
  onRecloned,
  onRenamed,
}: {
  jobId: string;
  speakers: SpeakerProfile[];
  /** Called after a re-clone starts, so the caller can start polling again. */
  onRecloned?: () => void;
  onRenamed?: (speakers: SpeakerProfile[]) => void;
}) {
  const [open, setOpen] = useState<string | null>(null);
  const [windows, setWindows] = useState<ReferenceWindow[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [alias, setAlias] = useState("");

  if (!speakers.length) return null;

  const flagged = speakers.filter((s) => s.status !== "cloned");

  async function openPicker(label: string) {
    if (open === label) {
      setOpen(null);
      return;
    }
    setOpen(label);
    setWindows(null);
    setError(null);
    try {
      // Scoped to this speaker: the picker must never offer another actor's
      // windows, or a re-clone could silently swap one voice for another.
      const r = await getReferenceWindows(jobId, label);
      setWindows(r.candidates);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not list windows");
      setWindows([]);
    }
  }

  async function apply(label: string, w: ReferenceWindow) {
    setBusy(true);
    setError(null);
    try {
      await chooseReference(jobId, w.start, w.duration, label);
      setOpen(null);
      onRecloned?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not change the reference");
    } finally {
      setBusy(false);
    }
  }

  async function saveAlias(label: string) {
    setBusy(true);
    try {
      const r = await renameSpeaker(jobId, label, alias);
      setEditing(null);
      onRenamed?.(r.speakers);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not rename");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <Sub>Speakers</Sub>

      {flagged.length > 0 && (
        <div className="mx-2.5 mb-1.5 rounded-[2px] border border-c-warn/40 bg-c-warn/5 px-2 py-1.5">
          {flagged.map((s) => (
            <p key={s.label}
               className="text-[10px] leading-relaxed text-c-warn">
              {s.warning}
            </p>
          ))}
        </div>
      )}

      <ul className="px-2.5 pb-1.5">
        {speakers.map((s) => (
          <li key={s.label} className="mb-2 border-b border-c-rule pb-2 last:border-0">
            <div className="flex items-center gap-1.5">
              <Lamp state={tone(s.status)} />
              <User size={11} className="shrink-0 text-c-mute" />
              <span className="min-w-0 flex-1 truncate text-[11px] text-c-text">
                {s.name}
              </span>
              <span className={cx(
                "flex items-center gap-1 text-[10px]",
                s.status === "cloned" ? "text-c-good"
                  : s.status === "short_reference" ? "text-c-warn" : "text-c-bad"
              )}>
                <StatusIcon status={s.status} />
                {speakerStatusLabel(s.status)}
              </span>
            </div>

            {/* The stable id, whenever it differs from what is displayed. */}
            {s.alias && (
              <p className="pl-[18px] text-[10px] text-c-faint">{s.label}</p>
            )}

            <p className="pl-[18px] text-[10px] leading-relaxed text-c-dim">
              {s.segment_count} segment{s.segment_count === 1 ? "" : "s"} ·{" "}
              {seconds(s.total_speech)} speech
              {s.phrase_count != null && ` · ${s.phrase_count} dubbed`}
            </p>

            {s.reference ? (
              <p className="pl-[18px] text-[10px] leading-relaxed text-c-dim">
                Reference: {seconds(s.reference.duration)}
                {s.reference.start != null && s.reference.end != null &&
                  ` · ${seconds(s.reference.start)}–${seconds(s.reference.end)}`}
                {s.reference_score != null &&
                  ` · score ${Number(s.reference_score).toFixed(2)}`}
              </p>
            ) : (
              <p className="pl-[18px] text-[10px] text-c-bad">
                No usable reference
              </p>
            )}

            <div className="mt-1 flex gap-1.5 pl-[18px]">
              {s.reference && (
                <audio
                  className="h-[22px] min-w-0 flex-1"
                  controls
                  preload="none"
                  src={artifactUrl(jobId, `speaker:${s.label}`)}
                  aria-label={`${s.name} reference preview`}
                />
              )}
              <Tool onClick={() => openPicker(s.label)} className="shrink-0">
                {open === s.label ? "Close" : "Change"}
              </Tool>
              <Tool
                className="shrink-0"
                onClick={() => {
                  setEditing(editing === s.label ? null : s.label);
                  setAlias(s.alias ?? "");
                }}
              >
                Name
              </Tool>
            </div>

            {editing === s.label && (
              <div className="mt-1 flex gap-1.5 pl-[18px]">
                <input
                  className="min-w-0 flex-1 rounded-[2px] border border-c-rule bg-c-input px-1.5 py-0.5 text-[10px] text-c-text"
                  value={alias}
                  placeholder={s.label}
                  onChange={(e) => setAlias(e.target.value)}
                  aria-label={`Display name for ${s.label}`}
                />
                <Tool primary disabled={busy}
                      onClick={() => saveAlias(s.label)}>Save</Tool>
              </div>
            )}

            {open === s.label && (
              <div className="mt-1 pl-[18px]">
                <p className="text-[10px] leading-relaxed text-c-faint">
                  Only {s.label}&apos;s own windows are listed. Choosing one
                  re-speaks this speaker&apos;s phrases and leaves the others
                  untouched.
                </p>
                {windows == null && (
                  <p className="text-[10px] text-c-mute">Loading…</p>
                )}
                {windows != null && windows.length === 0 && (
                  <p className="text-[10px] text-c-mute">
                    No alternative windows found for this speaker.
                  </p>
                )}
                {windows != null && windows.map((w) => (
                  <div key={`${w.start}-${w.duration}`}
                       className="mt-1 flex items-center gap-1.5">
                    <span className="min-w-0 flex-1 truncate text-[10px] text-c-dim">
                      {seconds(w.start)} · {seconds(w.duration)} · score{" "}
                      {Number(w.score).toFixed(2)}
                    </span>
                    <Tool disabled={busy} onClick={() => apply(s.label, w)}>
                      Use
                    </Tool>
                  </div>
                ))}
              </div>
            )}
          </li>
        ))}
      </ul>

      {error && (
        <p className="px-2.5 pb-1.5 text-[10px] leading-relaxed text-c-bad">
          {error}
        </p>
      )}
    </>
  );
}

/** The waveform-ish placeholder used when a job has no speakers to show. */
export function NoSpeakers() {
  return (
    <Row label="Speakers">
      <span className="flex items-center gap-1 text-[10px] text-c-mute">
        <Waveform size={11} />
        One voice
      </span>
    </Row>
  );
}
